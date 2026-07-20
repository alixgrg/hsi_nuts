# 03_pca_exploration_selection.ipynb

## Purpose

This notebook explores PCA behavior across matrix families and preprocessing chains, then writes the preprocessing shortlist consumed by SIMCA notebooks. It compares object-level and pixel-level matrix representations, scores PCA separability and stability metrics, and selects preprocessing methods only.

The notebook must not perform final model selection. Its output is a preprocessing shortlist, with a strict maximum of five rows for `object_matrix` and five rows for `pixel_matrix`.

## Main Inputs

- `HSI Data/processed/nir_uco_database.h5`
- Shared experiment configuration: `src/experiment_config.py`
- Pure reference objects for `REFERENCE_CLASSES`
- PCA batch subset: `PCA_ALLOWED_BATCHES = [1, 2, 3]`
- PCA scoring configuration built by `src.experiment_config.make_pca_selection_config()`
- Preprocessing methods listed in `PREPROCESSING_METHODS`

## Main Outputs

- `results/03_pca_<RESULTS_TAG>/pca_summary.parquet`
- `results/03_pca_<RESULTS_TAG>/pca_scoring_diagnostics.parquet`
- `results/03_pca_<RESULTS_TAG>/pca_selected_preprocessings.parquet`
- PCA score, loading, diagnostic, ranking, and tradeoff plots displayed in the notebook

## Execution Logic

1. Detect project root and import local modules.
2. Load the canonical HDF5 database.
3. Optionally restrict wavelengths with a configured wavelength window.
4. Filter to pure almond/peanut reference objects and batches `[1, 2, 3]`.
5. Build PCA-ready matrices for object and pixel matrix families.
6. Apply each preprocessing chain with `SpectralPreprocessor` through the PCA workflow.
7. Fit PCA models and compute diagnostic metrics:
   - explained variance
   - class separation
   - batch sensitivity
   - object-level separation for pixel matrices
   - projection shift and reconstruction diagnostics
8. Build the canonical PCA scoring configuration from `src.experiment_config`.
9. Score each PCA representation with `add_pca_selection_scores()`.
10. Build a separate score-diagnostic table with `build_pca_scoring_diagnostics()`.
11. Save the complete scored table to `pca_summary.parquet`.
12. Save the compact scoring diagnostic table to `pca_scoring_diagnostics.parquet`.
13. Build the strict preprocessing shortlist:
    - keep the best matrix variant for each `(matrix_family, preprocessing)` pair
    - rank candidates within each `matrix_family`
    - keep at most five preprocessing rows per family
    - keep justification columns: `selection_score`, `rank`, `matrix_family`, `matrix_variant`, `selection_reason`
14. Save the shortlist to `pca_selected_preprocessings.parquet`.
15. Re-read the Parquet shortlist and block execution if any family exceeds five rows.

## Selection Score Logic

The PCA selection score is a configurable multi-criteria score implemented in `src.workflows.pca_selection`. The project-level parameters are centralized in `src.experiment_config`, and notebook 03 builds its scoring object with `expcfg.make_pca_selection_config()`.

It is not a SIMCA performance metric and must not be interpreted as final model quality. It is used only to rank preprocessing candidates before SIMCA validation.

The score is family-specific:

- `object_matrix_score`: active for object-level matrices such as `object_mean` and `object_median`
- `pixel_matrix_score`: active for pixel-level matrices such as `balanced_pixels` and `all_pixels`
- `selection_score`: backward-compatible active score used for ranking, equal to the active family score after stability penalty

For object matrices, the default score rewards class separation and penalizes batch/projection instability:

- positive: `class_trace_ratio`, `mahalanobis_pc1_pc2_pc3`
- negative: `batch_trace_ratio`, `mean_train_projection_shift_norm`, `projection_q_deviation`, `ncomp_95`

For pixel matrices, the default score rewards object-aware class separation and penalizes object/batch instability:

- positive: `object_class_trace_ratio`, `object_over_intra_ratio`
- negative: `object_batch_trace_ratio`, `mean_intra_object_trace`, `mean_train_projection_shift_norm`, `projection_q_deviation`, `ncomp_95`

All metrics are robustly scaled within the configured comparison group, currently `matrix_variant`, using clipped quantiles, median, and IQR. The raw score is stored in `selection_score_without_stability`.

The final `selection_score` also includes a stability penalty:

- bootstrap resampling estimates how sensitive the score is to the candidate set used for robust scaling
- `selection_score_stability_std` stores the estimated score standard deviation
- `selection_score_rank_std` stores the estimated rank variability
- `selection_score_stability_penalty` is subtracted from the raw score

The notebook also writes relative quality diagnostics:

- `selection_flag`: relative flag such as `candidate`, `batch_sensitive`, `unstable_projection`, `high_projection_shift`, or `score_unstable`
- `pca_validation_warning`: non-blocking downstream-validation warning based on projection shift, Q residual deviation, and score stability
- `pca_validation_pass`: boolean convenience flag

These flags use distribution-relative thresholds by matrix family rather than fixed hard-coded thresholds, so they adapt better to new data.

The diagnostic scoring table is intentionally separate from the full PCA summary. `pca_summary.parquet` keeps all PCA metrics and artifacts needed downstream, while `pca_scoring_diagnostics.parquet` focuses on score inputs, metric contributions, stability penalties, thresholds, and warning flags for review.

To change the scoring behavior, edit `src/experiment_config.py` rather than notebook 03. The relevant project parameters are:

- `PCA_SELECTION_PROFILES`
- `PCA_SELECTION_EXPECTED_FAMILIES`
- `PCA_SELECTION_GROUP_COLS`
- `PCA_SELECTION_ROBUST_SCALING`
- `PCA_SELECTION_CLIP_QUANTILES`
- `PCA_SELECTION_QUALITY_LOWER_QUANTILE`
- `PCA_SELECTION_QUALITY_UPPER_QUANTILE`
- `PCA_SELECTION_VALIDATION_UPPER_QUANTILE`
- `PCA_SELECTION_BOOTSTRAP_ITERATIONS`
- `PCA_SELECTION_STABILITY_PENALTY_WEIGHT`
- `MAX_PCA_PREPROCESSINGS_PER_FAMILY`

## How To Use

1. Run notebooks 00, 01, and 02 first.
2. Keep `PCA_ALLOWED_BATCHES` equal to `[1, 2, 3]`.
3. Keep `RUN_ALL_PIXELS=False` unless the machine can handle the full pixel matrix cost.
4. Adjust `PREPROCESSING_METHODS` only when intentionally adding or removing preprocessing candidates.
5. Adjust PCA scoring only in `src/experiment_config.py`.
6. Run the notebook from top to bottom.
7. Before running notebook 04A, check that `pca_selected_preprocessings.parquet` has no more than five rows for each matrix family.

## Strict Selection Contract

The shortlist is intentionally family-specific:

- `object_matrix`: maximum 5 preprocessing rows
- `pixel_matrix`: maximum 5 preprocessing rows

The selected table keeps both direct justification columns and backward-compatible aliases:

- `selection_score`: stability-penalized PCA selection score used for ranking
- `selection_score_without_stability`: raw score before bootstrap stability penalty
- `selection_score_stability_std`: bootstrap score sensitivity estimate
- `rank`: global PCA rank from the scored table
- `matrix_family`: matrix family, usually `object_matrix` or `pixel_matrix`
- `matrix_variant`: best variant supporting the selected preprocessing
- `selection_reason`: human-readable reason containing the family, best variant, PCA quality flag, validation warning, and score stability
- `best_selection_score`, `best_rank`, `best_matrix_variant`, `selection_reasons`: compatibility aliases for downstream review

The notebook raises `RuntimeError` if:

- the in-memory shortlist is empty
- any family exceeds `MAX_PREPROCESSINGS_PER_MATRIX_FAMILY`
- either expected family is missing
- the saved Parquet file violates the same family limit after reloading

## Associated Modules And Functions

### `src.io.database_h5`

- `load_nir_uco_h5(path, reconstruct_heavy_object_arrays=True)`: loads object and image records for PCA exploration.

### `src.matrices.matrix_registry`

- `build_matrix(...)`: builds object and pixel matrix representations from filtered object records.

### `src.spectra.preprocessing_configs`

- `normalize_preprocessing_configs(...)`: validates and normalizes preprocessing aliases and explicit step chains.

### `src.spectra.band_selection`

- `select_wavelength_range_from_database(...)`: applies optional wavelength slicing to both object and image records.
- `wavelength_selection_summary(info)`: creates a compact wavelength-selection table.

### `src.workflows.pca`

- `compare_pca_representations(...)`: main workflow function that builds matrices, applies preprocessing, fits PCA, and returns summary metrics plus detailed PCA artifacts.
- `pca_matrix_family_from_method(...)`: maps matrix methods to `object_matrix` or `pixel_matrix`.
- `pca_matrix_variant_from_method(...)`: creates stable labels such as `object_mean`, `object_median`, `balanced_pixels_random`, and `balanced_pixels_center`.
- `binary_class_separation_scores(...)`: computes quick class-separation metrics in PCA score space.
- `mahalanobis_centroid_distance(...)`: measures class centroid distance with covariance regularization.
- `trace_ratio_by_group(...)`: compares between-group to within-group variance in PCA space.
- `pca_distance_summary(...)`: summarizes Hotelling T2 and Q residual diagnostics.
- `train_projection_shift_by_label(...)`: measures train/projection shift by class.
- `pixel_object_score_metrics(...)`: computes object-aware metrics for pixel matrices.
- `compute_pca_summary_metrics(...)`: assembles PCA diagnostic metrics for one representation.
- `add_pca_selection_score(...)`: deprecated compatibility wrapper that delegates to `src.workflows.pca_selection`; new code should use `add_pca_selection_scores(...)`.

### `src.workflows.pca_selection`

- `PCASelectionProfile`: stores metric weights and diagnostic metric names for one matrix family.
- `PCASelectionConfig`: stores score configuration, grouping, robust scaling, stability bootstrap, quality thresholds, and shortlist limits.
- `make_pca_selection_config(...)`: lower-level builder used by `src.experiment_config` to create a `PCASelectionConfig`.
- `add_pca_selection_scores(...)`: computes family-specific scores, active `selection_score`, bootstrap stability diagnostics, and relative validation flags.
- `build_pca_scoring_diagnostics(...)`: extracts the compact scoring-review table saved as `pca_scoring_diagnostics.parquet`.
- `add_pca_relative_quality_flags(...)`: adds distribution-relative quality and validation warnings.
- `format_pca_selection_reason(...)`: builds the human-readable reason stored in the selected shortlist.
- `select_pca_preprocessing_shortlist(...)`: keeps the best variant per `(matrix_family, preprocessing)` and selects at most five preprocessing rows per matrix family.
- `validate_pca_preprocessing_shortlist(...)`: blocking control used before and after writing the shortlist Parquet file.

### `src.visualization.plot_pca`

- `plot_explained_variance(...)`: shows PCA variance explained by component.
- `plot_loadings(...)`: displays PCA loading vectors.
- `plot_pca_diagnostic(...)`: plots Q residuals and Hotelling T2 diagnostics.
- `plot_pca_metric_heatmap(...)`: compares PCA metrics across preprocessing and matrix variants.
- `plot_pca_metric_tradeoff(...)`: visualizes tradeoffs such as separation versus batch sensitivity.
- `plot_pca_metric_ranking(...)`: displays ranked preprocessing and matrix candidates.

### `src.visualization.plot_scores`

- `plot_scores(...)`: plots PCA scores for selected components.
- `build_scores_dataframe(...)`: converts score arrays and metadata into a plotting table.
- `sample_scores_dataframe(...)`: samples large score tables for plotting.
- `plot_scores_density(...)`: displays score density by group.
- `summarize_scores_by_object(...)`: aggregates pixel-level scores to objects.
- `plot_object_score_summary(...)`: visualizes object-level score summaries.

### `src.visualization.plot_spectra`

- `plot_spectra(...)`: displays representative raw or preprocessed spectra.

### `src.utils`

- `save_parquet(...)`: saves `pca_summary.parquet` and `pca_selected_preprocessings.parquet`.

### `src.experiment_config`

- Provides shared target classes, PCA batches, random state, balanced-pixel settings, PCA shortlist limits, PCA scoring weights, scoring quantiles, bootstrap settings, and `make_pca_selection_config()`.

## Maintenance Checks

- `PCA_ALLOWED_BATCHES` must remain `[1, 2, 3]` while batch 4 is reserved for pure-test evaluation.
- The shortlist must stay preprocessing-only; model selection belongs to notebook 05.
- `pca_summary.parquet` should retain all evaluated PCA combinations.
- `pca_scoring_diagnostics.parquet` should be used for scoring review and troubleshooting, not as the downstream modeling input.
- `pca_selected_preprocessings.parquet` must have no more than five rows per matrix family.
- Both `object_matrix` and `pixel_matrix` should be present unless the experiment protocol is explicitly changed.

## Automated Tests

- `tests/test_pca_selection.py` covers family-specific PCA scoring, bootstrap stability penalties, strict shortlist limits, canonical config creation, and the separate diagnostic scoring table.
- The tests also verify that the legacy `src.workflows.pca.add_pca_selection_score()` wrapper delegates to the official selection module instead of carrying independent scoring logic.
