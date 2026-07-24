# Notebook 03 — PCA Exploration and Preprocessing Selection

Documentation for `03_pca_exploration_selection.ipynb`.

## Purpose

This notebook compares Principal Component Analysis representations for the NIR UCO peanut-detection project and creates a preprocessing shortlist for downstream SIMCA and MCR analyses.

It evaluates:

- object-mean spectra;
- object-median spectra;
- balanced pixel spectra selected randomly;
- balanced pixel spectra selected near object centroids;
- optionally, all object pixels;
- 18 spectral preprocessing chains;
- class separation, batch effects, explained variance, PCA distances, and object-level pixel diagnostics;
- a family-specific, robustly scaled selection score with a bootstrap-derived stability penalty.

The notebook saves a complete scored PCA summary, a compact scoring-diagnostics table, and a strict size-limited preprocessing shortlist for the next modelling stage.

> **Scope:** this is an exploratory representation and preprocessing-selection notebook. It does not select the final SIMCA model, estimate final predictive performance, define anomaly thresholds, or evaluate mixture detection.

> **Important:** the reported PCA metrics are descriptive and in-sample. The same pure objects are used to fit preprocessing, fit PCA, and calculate separation/batch diagnostics. The held-out pure batch 4 is excluded but is not evaluated in this notebook.

## Position in the workflow

```text
00_building_database.ipynb
    -> nir_uco_database.h5
01_database_quality_check.ipynb
    -> database QC
02_matrices_preprocessing.ipynb
    -> matrix and preprocessing compatibility
03_pca_exploration_selection.ipynb
    -> pca_summary.parquet
    -> pca_scoring_diagnostics.parquet
    -> pca_selected_preprocessings.parquet
04A_simca_grid_validation.ipynb
    -> model validation
```

Notebook 03 loads the HDF5 database directly. It does not enforce the QC results from Notebook 01 or read the matrix/preprocessing summaries from Notebook 02. The user must ensure that those earlier checks have been completed and accepted.

## Processing overview

The notebook performs the following operations:

1. Locate the project root and import the PCA workflow and visualization modules.
2. Configure the spectral range, object subset, PCA size, matrix variants, preprocessing chains, selection policy, and output paths.
3. Load the HDF5 image and object databases.
4. Optionally restrict spectral arrays to a wavelength window.
5. Filter to pure almond and peanut objects from batches 1–3.
6. Inspect raw object-mean spectra.
7. Build four PCA matrix variants by default.
8. Fit 18 preprocessing/PCA combinations for every matrix variant.
9. Calculate PCA variance, class, batch, distance, and object-level diagnostics.
10. Robustly scale metrics and calculate a family-specific selection score.
11. Estimate score sensitivity by bootstrap resampling of the candidate reference distribution.
12. Apply relative warning flags.
13. Save the scored summary and scoring diagnostics.
14. Inspect rankings, heatmaps, Pareto trade-offs, scores, loadings, and Q/T² plots.
15. Deduplicate preprocessing names within each matrix family and select at most five per family.
16. Validate the shortlist before and after Parquet serialization.

## Expected project layout

```text
project_root/
├── notebooks/
│   ├── 00_building_database.ipynb
│   ├── 01_database_quality_check.ipynb
│   ├── 02_matrices_preprocessing.ipynb
│   └── 03_pca_exploration_selection.ipynb
├── src/
│   ├── experiment_config.py
│   ├── utils.py
│   ├── io/
│   │   └── database_h5.py
│   ├── matrices/
│   │   ├── matrix_registry.py
│   │   └── redim_matrix.py
│   ├── models/
│   │   └── pca.py
│   ├── spectra/
│   │   ├── band_selection.py
│   │   ├── preprocessing.py
│   │   └── preprocessing_configs.py
│   ├── workflows/
│   │   ├── pca.py
│   │   └── pca_selection.py
│   └── visualization/
│       ├── plot_pca.py
│       ├── plot_scores.py
│       └── plot_spectra.py
├── HSI Data/
│   └── processed/
│       └── nir_uco_database.h5
└── results/
```

The two PCA modules have distinct roles:

- `src/models/pca.py` implements covariance-based PCA, projection, reconstruction, Q residuals, and Hotelling T²;
- `src/workflows/pca.py` builds matrices, applies preprocessing, fits models, and produces comparison metrics.

The notebook creates a tagged result directory such as `results/03_pca_non_noisy_all/`.

## Requirements

The recorded kernel is named `hsi-nuts`. Use the same pinned project environment as the preceding notebooks. Relevant packages include:

- `numpy`
- `pandas`
- `scipy`
- `h5py`
- `plotly`
- a pandas-compatible Parquet engine, normally `pyarrow`
- JupyterLab or Jupyter Notebook

A minimal installation command is:

```bash
python -m pip install numpy pandas scipy h5py plotly pyarrow jupyterlab
```

## How to run the notebook

1. Complete the database construction, QC, and matrix/preprocessing checks in Notebooks 00–02.

2. Verify the database path:

   ```text
   HSI Data/processed/nir_uco_database.h5
   ```

3. Activate the project environment.

4. Start Jupyter from the project root or its direct `notebooks/` directory:

   ```bash
   jupyter lab notebooks/03_pca_exploration_selection.ipynb
   ```

5. Review the allowed batches, matrix variants, preprocessing chains, PCA component count, and selection configuration.

6. Restart the kernel and run all cells in order.

7. Inspect both the numerical ranking and the diagnostic plots. Do not treat the global top row as automatically accepted when it carries a warning flag.

8. Verify the saved shortlist family counts before using it in Notebook 04A.

The notebook uses `%autoreload 2`. Restart and rerun after modifying PCA, preprocessing, matrix, or selection code.

## Configuration reference

### Input and wavelength settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `DB_H5_PATH` | `HSI Data/processed/nir_uco_database.h5` | Database built by Notebook 00. |
| `USE_WAVELENGTH_WINDOW` | `False` | Use every stored non-noisy band by default. |
| `WAVELENGTH_MODE` | `non_noisy_all` | Descriptive project wavelength mode. |
| `WINDOW_MIN_NM` | `1225.0` | Optional requested lower wavelength bound. |
| `WINDOW_MAX_NM` | `1675.0` | Optional requested upper wavelength bound. |
| `RESULTS_TAG` | `non_noisy_all` | Output-directory suffix when no window is used. |

The recorded run uses 63 bands from approximately 960.74 to 1702.00 nm. Enabling the inclusive 1225–1675 nm window on this axis would retain the 37 available bands from approximately 1235.72 to 1666.13 nm.

As in Notebook 02, `WAVELENGTH_MODE` remains the default label when a custom window is enabled. Interpret it together with `use_wavelength_window`, requested bounds, actual endpoints, and `results_tag`.

### PCA object subset

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `TARGET_CLASS` | `peanut` | Project anomaly/target class. |
| `REFERENCE_CLASSES` | `almond`, `peanut` | Classes used for exploratory PCA. |
| `PCA_SAMPLE_KIND` | `pure` | Excludes mixtures and position-reference images. |
| `PCA_ALLOWED_BATCHES` | `[1, 2, 3]` | Training and validation batches used for selection. |

The notebook raises `ValueError` if `PCA_ALLOWED_BATCHES` differs from exactly `[1, 2, 3]`. Pure batch 4 is reserved as a later test batch.

The object filter ignores the stored `split` field because Notebook 00 assigned `projection` to every object. Selection is based directly on sample kind, object class, and batch.

### PCA and matrix settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `N_COMPONENTS` | `20` | Number of loadings/scores retained by each fitted PCA model. |
| `M_BALANCED_PIXELS` | `40` | Requested pixels per object. |
| `REPLACE_BALANCED_PIXELS` | `False` | Small objects contribute all available pixels rather than duplicates. |
| `RANDOM_STATE` | `42` | Seed for pixel selection, spectrum display sampling, and score bootstrap. |
| `BALANCED_PIXEL_STRATEGIES` | `random`, `center` | Pixel-selection variants. |
| `RUN_ALL_PIXELS` | `False` | Excludes the potentially large all-pixel matrix by default. |

Default matrix variants are:

```text
object_mean
object_median
balanced_pixels_random
balanced_pixels_center
```

Enabling `RUN_ALL_PIXELS=True` adds `all_pixels`.

Balanced-pixel semantics and their sampling caveats are described in the Notebook 02 documentation. In particular, the current random helper resets the same seed for every object, so objects with equal pixel counts can receive identical relative index patterns.

### Preprocessing settings

The notebook evaluates 18 configurations:

```text
raw
absorbance
snv
msc
sg_smooth
sg_d1
sg_d2
absorbance + snv
absorbance + msc
absorbance + sg_smooth
absorbance + sg_d1
absorbance + sg_d2
snv + sg_smooth
snv + sg_d1
snv + sg_d2
absorbance + snv + sg_smooth
absorbance + snv + sg_d1
absorbance + snv + sg_d2
```

| Parameter | Current value |
| --- | ---: |
| `SG_WINDOW_LENGTH` | `11` |
| `SG_POLYORDER` | `2` |

The preprocessor automatically raises the effective polynomial order to at least 3 for `sg_d2`, although the notebook protocol records the configured value 2.

Absorbance conversion clips nonpositive reflectance to the preprocessor epsilon instead of flagging it. Balanced-pixel matrices can contain negative raw reflectance, so absorbance-based pixel PCA results require physical review.

MSC is fitted independently on the complete matrix for each representation. This is suitable for descriptive exploration but must be fitted on training data only during predictive validation.

### Plot settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `N_TOP_TO_DISPLAY` | `20` | Number of top scoring rows displayed. |
| `N_TOP_TO_PLOT` | `5` | Maximum detailed candidate rows. |
| `RUN_SPECTRA_CHECK_PLOTS` | `True` | Raw spectrum plots in Cell 8. |
| `RUN_DETAILED_PCA_PLOTS` | `True` | Explained variance, scores, loadings, and Q/T² plots in Cells 32–36. |
| `MAX_SPECTRA_TO_PLOT` | `50` | Approximate total individual raw spectra displayed. |

Ranking plots, heatmaps, and trade-off plots in Cells 21–26 run regardless of `RUN_DETAILED_PCA_PLOTS`.

## Covariance PCA implementation

`PCAModel.fit`:

1. converts the input to a two-dimensional float array;
2. subtracts the column-wise mean because `center=True`;
3. computes the covariance matrix as `X_centered.T @ X_centered / (N - 1)`;
4. performs symmetric eigendecomposition with `numpy.linalg.eigh`;
5. sorts eigenvalues and eigenvectors in descending eigenvalue order;
6. retains the first `N_COMPONENTS` loading vectors;
7. calculates scores as `X_centered @ loadings`.

The PCA model centers features but does not automatically standardize spectral bands. Any scaling comes from the selected spectral preprocessing.

The model provides:

- `transform` and `inverse_transform`;
- reconstruction with a chosen component count;
- Q residuals, calculated as squared reconstruction-error norm;
- Hotelling T², calculated as the sum of squared retained scores divided by corresponding eigenvalues.

Small eigenvalues used by T² are lower-bounded by `eps` for numerical stability.

Although only 20 loadings are retained, explained-variance and cumulative-variance arrays contain all covariance eigenvalues. Therefore, `ncomp_90` or `ncomp_95` can exceed 20. The recorded summary includes examples with `ncomp_95=21`, even though those fitted models retain only 20 score components. Such a model must be refitted with more components if downstream analysis needs the stated variance threshold.

## PCA comparison metrics

Most diagnostics use the first three principal components, even though each model retains 20.

### Explained variance

| Metric | Meaning |
| --- | --- |
| `evr_pc1`, `evr_pc2`, `evr_pc3` | Individual explained-variance ratios. |
| `cum_pc2`, `cum_pc3` | Cumulative variance through PC2 or PC3. |
| `ncomp_90`, `ncomp_95` | Smallest component count reaching 90% or 95% based on the full eigenvalue spectrum. |

### Two-class score-space separation

For almond and peanut:

- `centroid_distance_pc1_pc2` is the Euclidean distance between class centroids in PC1–PC2;
- `fisher_pc1`, `fisher_pc2`, and `fisher_pc3` use squared mean difference divided by the sum of within-class variances on each axis;
- `mahalanobis_pc1_pc2` and `mahalanobis_pc1_pc2_pc3` use the regularized pooled within-class covariance.

These are in-sample geometric diagnostics, not cross-validated classification scores.

### Class and batch trace ratios

For the first three score columns, `trace_ratio_by_group` computes:

```text
between-group sum of squares / within-group sum of squares
```

The notebook reports:

- `class_trace_ratio` for almond versus peanut;
- `batch_trace_ratio` for batches 1, 2, and 3;
- `class_over_batch_ratio = class_trace_ratio / batch_trace_ratio`.

Higher class separation is desirable and higher batch separation is treated as undesirable. Batch and class effects can still be confounded if the acquisition design is unbalanced.

### Training Q and T² summaries

`train_q_mean`, `train_q_median`, `train_q_q95`, `train_t2_mean`, `train_t2_median`, and `train_t2_q95` are calculated on the same training matrix with three components.

No statistical control limits are derived here. The Q-versus-T² figure is exploratory and does not define outliers.

### Pixel-to-object diagnostics

For balanced or all-pixel matrices, pixel scores are averaged by object before calculating:

- `object_class_trace_ratio`;
- `object_batch_trace_ratio`.

The workflow also calculates:

- `mean_intra_object_trace`: mean covariance trace of pixel scores within objects;
- `object_over_intra_ratio`: total between-object score variance divided by mean within-object score variance.

These object-level metrics prevent raw pixel counts from being the only basis for selecting a pixel representation. The aggregation gives every object one centroid for class and batch trace ratios.

## Selection-score design

### Matrix families

Matrix methods are grouped into:

```text
object_matrix: object_mean, object_median
pixel_matrix:  balanced_pixels, all_pixels, pixel
```

Each family uses a different scoring profile.

### Object-matrix profile

| Direction | Metric | Weight |
| --- | --- | ---: |
| Reward | `class_trace_ratio` | `3.0` |
| Reward | `mahalanobis_pc1_pc2_pc3` | `1.0` |
| Penalize | `batch_trace_ratio` | `2.0` |
| Penalize | `mean_train_projection_shift_norm` | `1.5` |
| Penalize | `projection_q_deviation` | `1.5` |
| Penalize | `ncomp_95` | `0.3` |

### Pixel-matrix profile

| Direction | Metric | Weight |
| --- | --- | ---: |
| Reward | `object_class_trace_ratio` | `3.0` |
| Reward | `object_over_intra_ratio` | `1.0` |
| Penalize | `object_batch_trace_ratio` | `2.0` |
| Penalize | `mean_intra_object_trace` | `1.0` |
| Penalize | `mean_train_projection_shift_norm` | `1.2` |
| Penalize | `projection_q_deviation` | `1.2` |
| Penalize | `ncomp_95` | `0.3` |

`compare_pca_representations` does not pass a projection dataset in this notebook. Consequently, `mean_train_projection_shift_norm` and `projection_q_deviation` are absent and their contributions are set to zero. Projection-related flags cannot fire in the recorded workflow, despite being present in the configured profiles.

### Robust metric scaling

Scores are calculated independently within each `matrix_variant`, using its 18 preprocessing rows as the comparison group.

For every active metric:

1. replace infinities with missing values in the reference series;
2. clip values to the 5th and 95th percentiles;
3. subtract the clipped median;
4. divide by the clipped interquartile range;
5. multiply by the family-specific weight;
6. add rewards and subtract penalties.

If a metric is missing or its scale is effectively zero, its scaled contribution is zero.

Because scaling is relative within each variant, `selection_score` is not an absolute scientific quality measure. Global comparisons between matrix families should be interpreted more cautiously than within-family shortlisting.

### Bootstrap stability penalty

For each matrix variant, the selection code performs 100 bootstrap iterations. Each iteration resamples the 18 preprocessing candidate rows with replacement to form the reference distribution used for robust scaling, then rescales and re-ranks the original rows.

It records:

- mean bootstrapped score;
- standard deviation of bootstrapped scores;
- standard deviation of bootstrapped ranks.

The final score is:

```text
selection_score = selection_score_without_stability
                  - 0.25 * selection_score_stability_std
```

This procedure measures sensitivity of the scoring scale and ranking to the candidate reference set. It does **not** resample objects, batches, spectra, preprocessing fits, or PCA models. It must not be described as predictive bootstrap validation or PCA model stability.

### Relative quality flags

Thresholds are derived separately for the object and pixel families:

- weak separation: at or below the family 25th percentile;
- high batch effect: at or above the family 75th percentile;
- high projection diagnostic: at or above the family 75th percentile when present;
- high score instability: at or above the family 75th percentile.

Possible warnings are:

```text
weak_relative_separation
batch_sensitive
unstable_projection
high_projection_shift
score_unstable
```

`selection_flag` stores only the first triggered warning in the rule order, while `pca_validation_warning` joins all triggered warnings. `pca_validation_pass` is true only when no warning is present.

The flags are relative to the evaluated candidate pool. `candidate` means no relative warning fired; it is not final model approval.

## Shortlist construction

`select_pca_preprocessing_shortlist`:

1. sorts rows by family, preprocessing name, score, and global rank;
2. keeps the best matrix variant for each preprocessing name within each family;
3. adds the selected variant/method/strategy and a human-readable reason;
4. ranks the deduplicated candidates inside each family;
5. keeps at most five preprocessing names per family.

The shortlist is constrained to include both expected families and no more than five rows per family. It is validated before saving and again after reloading from Parquet.

The selection procedure does **not** filter on `pca_validation_pass`, `selection_flag`, or warning text. A flagged candidate can therefore enter the shortlist when its score is high. This occurs in the recorded run.

The validation function checks non-emptiness, family presence, and maximum family size. It does not require exactly five rows, require candidates to pass warnings, or revalidate every metric value.

## Detailed notebook walkthrough

Cell numbers are zero-based and correspond to positions in the `.ipynb` file.

### Cell 0 — Objectives

Introduces matrix/preprocessing comparison and the downstream shortlist. It correctly describes the notebook as exploratory rather than final SIMCA selection.

### Cell 1 — Imports and project-root detection

Imports NumPy, pandas, warnings, display utilities, and path handling; expands DataFrame display limits; finds `PROJECT_ROOT`; and adds it to `sys.path`.

The imported `warnings` module is not used later.

### Cell 2 — Project imports and autoreload

Imports database loading, matrix construction, preprocessing and band-selection helpers, PCA comparison/selection workflows, and all plotting utilities. `%autoreload 2` supports source-module development.

### Cell 3 — Configuration

Defines spectral mode, output paths, PCA subset, batch contract, component count, matrix variants, 18 preprocessing methods, selection config, and plotting switches.

The output directory is created immediately. There is no overwrite guard.

### Cell 4 — Database and wavelength handling

Loads the HDF5 database with heavy-array reconstruction enabled. It either applies the optional wavelength window or obtains the axis from the first object, then displays a one-row wavelength configuration.

Neither `wavelength_config_df` nor the detailed `wavelength_selection_df` is saved by this notebook.

The cell has no explicit database existence preflight and assumes a nonempty object database.

Heavy array reconstruction is not required for the matrices and PCA diagnostics used here and increases memory consumption.

### Cell 5 — Full object inventory

Builds a lightweight metadata table for all database objects and displays counts by sample kind, class, and batch. This table is not saved.

### Cell 6 — PCA subset

Defines a local filtering helper and selects pure almond and peanut objects from batches 1–3. It raises `RuntimeError` when selection is empty.

In the recorded run, the subset contains 317 objects: 166 almonds and 151 peanuts.

### Cell 7 — Wavelength display

Reads the wavelength axis from the first selected object and prints endpoints. This repeats part of Cell 4. If the axis is absent, plots may use band indices, but Cell 37 later assumes a wavelength array when locating maximum class differences.

### Cell 8 — Raw object-mean matrix and spectra

Builds a `(317, 63)` object-mean matrix. If plotting is enabled, pandas samples approximately half of `MAX_SPECTRA_TO_PLOT` per class with the same random seed, then displays individual spectra and class mean/standard-deviation curves.

A NumPy generator is created but not used; sampling is performed by `DataFrame.sample`.

### Cells 9–10 — PCA comparison design

Defines the default run groups:

- object mean and median together;
- balanced random;
- balanced center when configured;
- optional all pixels.

The object-database subset is already pure and batch-restricted, so split filtering is deliberately disabled.

### Cell 11 — Preprocessing validation

Normalizes and validates the 18 preprocessing configurations. Unknown steps or invalid `raw` chains stop execution.

### Cell 12 — Matrix/preprocessing PCA fits

For each run and matrix method, `compare_pca_representations` builds the matrix once, then for every preprocessing:

1. fits a new `SpectralPreprocessor` on the complete matrix;
2. transforms the same matrix;
3. fits a centered 20-component PCA;
4. calculates scores, loadings, variance, separation, batch, Q/T², and pixel-object diagnostics;
5. retains the full transformed matrix, model, preprocessor, metadata, and metrics in memory.

The loop catches errors at the run level, prints the failing run, and re-raises. One failing configuration stops the notebook; failures are not accumulated into a table.

Backward-compatibility code creates missing strategy/family/variant columns when an older workflow module is loaded. This preserves execution but does not guarantee that older code computed the same metrics.

The recorded unscored summary has 72 rows and 42 columns.

### Cell 13 — Selection scores and diagnostics

Adds family-specific scores, bootstrap stability, relative warning thresholds, flags, and contributions. Rows are sorted by penalized score and assigned global ranks. A compact scoring-diagnostics table is built separately.

### Cells 14–15 — Schema inspection

Displays all scored columns, prints pre- and post-scoring schemas, asserts that `matrix_variant` exists and is complete, and shows distinct family/method/strategy/variant combinations.

### Cell 16 — Save scored summaries

Writes:

- `pca_summary.parquet`, containing the full scored table;
- `pca_scoring_diagnostics.parquet`, containing score inputs, contributions, thresholds, stability, and flags.

### Cells 17–20 — Ranking tables

Defines the interpretation of the global ranking, displays the top 20 overall, and then separates object and pixel matrix rows.

The filter uses `matrix_method.str.startswith("object")` to identify object matrices. This works for the current names but couples family detection to a naming convention rather than the existing `matrix_family` field.

### Cells 21–22 — Ranking plots

Draw separate bar rankings for object and pixel candidates using `selection_score`.

### Cells 23–24 — Metric heatmaps

Display preprocessing-by-matrix-variant heatmaps for class and batch trace ratios.

### Cell 25 — Global class/batch trade-off

Plots batch trace ratio on the x-axis and class trace ratio on the y-axis. The optional Pareto front treats lower x and higher y as better. Labels are added to eight points.

### Cell 26 — Pixel object-level trade-off

Repeats the trade-off for pixel matrices using object-level class and batch trace ratios, with marker size representing `object_over_intra_ratio`.

### Cell 27 — Result lookup helpers

Defines a function for retrieving an in-memory result from `run_id`, matrix method, and preprocessing, plus a metadata fallback helper supporting old and new key names.

### Cells 28–30 — Best row per variant

Select the highest-scoring candidate for each pixel and object matrix variant, then concatenate them. `groupby(...).first()` takes the first row after score sorting.

### Cell 31 — Detailed candidate set

Starts with the global top candidate, adds the best row from every variant, deduplicates by run/method/preprocessing/strategy, and keeps at most five rows.

The recorded detailed set contains four rows—one per active matrix variant. Two of these rows carry warnings.

### Cell 32 — Explained variance plots

For every detailed row, displays individual and cumulative explained variance for up to 12 components, with 90% and 95% reference lines.

### Cell 33 — Class score plots

For object matrices, draws PC1/PC2 scatter plots colored by class and symbolized by batch.

For pixel matrices, builds a tidy score DataFrame and displays class density contours faceted by batch. Density plots reduce overplotting but can hide sparse local structure.

### Cell 34 — Batch and object score plots

Object matrices are plotted with batch colors/symbols. Pixel scores are aggregated by object; the plot uses mean PC1/PC2, class color, batch symbol, and selected pixel count as marker size.

For balanced matrices, `n_pixels` in this plot is the number of sampled rows retained for the object, not its full segmented area.

### Cell 35 — Loading plots

Plots PC1–PC3 loadings against wavelengths and adds explained-variance percentages to legend names. Loading signs are arbitrary; interpret peaks and shapes, not absolute sign orientation across independent PCA fits.

### Cell 36 — Q/T² plots

Displays training Q residuals against Hotelling T² using three components. No control limits or automated rejection decisions are calculated.

### Cell 37 — Mean class-difference table

For every in-memory result, calculates the mean preprocessed spectrum for peanut and almond, their feature-wise difference, mean/max absolute difference, and wavelength of maximum difference.

These values are not comparable across preprocessing methods with different units or scaling. Pixel matrices are also pixel-weighted. The table is displayed but not saved.

If no wavelength axis exists, the function fails while indexing `wavelengths`, despite the earlier fallback message about band indices.

### Cells 38–39 — Interpretation columns

Defines a compact set of metrics intended for interpretation and later shortlist displays. Cell 39 creates the available-column list but does not itself display a table.

### Cell 40 — Strict preprocessing shortlist

Builds the deduplicated candidate pool and selects at most five preprocessing names per matrix family. It displays candidates, selected rows, counts, variants, reasons, flags, and warnings.

### Cell 41 — Shortlist validation and save

Validates the family-size contract, saves `pca_selected_preprocessings.parquet`, reads it back, and validates again. The recorded result contains five object-family and five pixel-family rows.

### Cell 42 — In-memory PCA protocol

Builds a one-row protocol DataFrame containing database path, wavelength settings, subset, component count, sampling, preprocessing, scoring, stability, and shortlist counts.

The protocol is displayed but not saved. It also records requested `sg_polyorder=2`, not the effective order 3 used by `sg_d2`.

### Cell 43 — Completion summary

Lists the three essential files, data/configuration sizes, global top candidate, and the next notebook.

## Persisted outputs

| Output | Purpose |
| --- | --- |
| `pca_summary.parquet` | Full 72-row scored PCA comparison table. |
| `pca_scoring_diagnostics.parquet` | Compact scoring inputs, contributions, stability measures, thresholds, and flags. |
| `pca_selected_preprocessings.parquet` | Ten-row family-constrained shortlist for downstream modelling. |

For the recorded run, all files are written under:

```text
results/03_pca_non_noisy_all/
```

The notebook does not save:

- fitted PCA models;
- fitted spectral preprocessors or MSC references;
- PCA scores, loadings, or transformed matrices;
- wavelength configuration or selected wavelength indices;
- full object inventory;
- mean class-difference results;
- the one-row PCA protocol;
- Plotly figures.

All three required Parquet files are overwritten on rerun; there is no overwrite guard or run versioning.

## Output table schemas

### `pca_summary.parquet`

The scored table contains the original PCA metrics plus selection metadata. Major groups include:

| Group | Representative columns |
| --- | --- |
| Identity | `rank`, `run_id`, `matrix_family`, `matrix_variant`, `matrix_method`, strategies, `preprocessing`, `preprocessing_steps` |
| Data shape | `n_observations`, `n_bands`, `n_components`, `m`, label counts |
| Variance | `evr_pc1`–`evr_pc3`, `cum_pc2`, `cum_pc3`, `ncomp_90`, `ncomp_95` |
| Class separation | centroid, Fisher, Mahalanobis, `class_trace_ratio` |
| Batch | `batch_trace_ratio`, `class_over_batch_ratio` |
| PCA distance | training Q and T² mean/median/95th percentile |
| Pixel-object metrics | object class/batch ratios, intra-object trace, object/intra ratio |
| Scoring | raw family scores, metric contributions, score before/after stability penalty |
| Stability | bootstrapped score/rank mean or standard deviations and penalty |
| Relative QC | thresholds, `selection_flag`, full warning, pass Boolean |

The `label_counts` dictionary is converted to a Parquet-safe representation by the shared export helper.

### `pca_scoring_diagnostics.parquet`

Contains only identifiers, active metrics, contribution columns, raw/final scores, bootstrap diagnostics, thresholds, and flags. Use this table to audit why one candidate outranked another without loading every PCA descriptive metric.

### `pca_selected_preprocessings.parquet`

The shortlist inherits scored metrics for each chosen row and adds fields such as:

```text
family_selection_rank
selection_reason
best_selection_score
best_rank
best_matrix_variant
selected_from_variants
selected_from_methods
selected_from_strategies
selection_reasons
best_selection_flag
```

Each row represents one preprocessing name within one matrix family, using its best-scoring matrix variant.

## Reference run recorded in the notebook

### Data and run size

| Metric | Recorded value |
| --- | ---: |
| Images loaded | 48 |
| Objects in full database | 1,262 |
| PCA pure-object subset | 317 |
| Almond objects | 166 |
| Peanut objects | 151 |
| Allowed batches | 1, 2, 3 |
| Held-out pure batch | 4 |
| Active bands | 63 |
| PCA components retained | 20 |
| Preprocessing methods | 18 |
| Matrix variants | 4 |
| PCA combinations | 72 |

Matrix sizes were:

| Variant | Shape | Class rows |
| --- | ---: | --- |
| `object_mean` | `(317, 63)` | almond 166, peanut 151 |
| `object_median` | `(317, 63)` | almond 166, peanut 151 |
| `balanced_pixels_random` | `(12360, 63)` | almond 6,494, peanut 5,866 |
| `balanced_pixels_center` | `(12360, 63)` | almond 6,494, peanut 5,866 |

### Global top candidate

The highest penalized score is:

| Field | Recorded value |
| --- | --- |
| Matrix variant | `balanced_pixels_random` |
| Preprocessing | `snv_sg_smooth` |
| Selection score | `5.823665` |
| Warning flag | `score_unstable` |
| Score before stability penalty | `6.874581` |
| Stability standard deviation | `4.203665` |
| `ncomp_90` / `ncomp_95` | 4 / 4 |
| Cumulative variance through PC3 | `0.873493` |
| Object class trace ratio | `0.192140` |
| Object batch trace ratio | `0.027821` |

Its high rank does not override the instability warning.

### Saved shortlist

| Family | Family rank | Preprocessing | Best variant | Score | Flag |
| --- | ---: | --- | --- | ---: | --- |
| Object | 1 | `absorbance_sg_d1` | `object_mean` | 2.568244 | `candidate` |
| Object | 2 | `absorbance_sg_d2` | `object_mean` | 2.169377 | `weak_relative_separation` |
| Object | 3 | `absorbance_snv_sg_d2` | `object_median` | 1.964249 | `candidate` |
| Object | 4 | `snv_sg_d2` | `object_median` | 1.300178 | `candidate` |
| Object | 5 | `absorbance_snv_sg_d1` | `object_mean` | 0.932149 | `candidate` |
| Pixel | 1 | `snv_sg_smooth` | `balanced_pixels_random` | 5.823665 | `score_unstable` |
| Pixel | 2 | `absorbance_snv_sg_smooth` | `balanced_pixels_random` | 5.254788 | `candidate` |
| Pixel | 3 | `snv` | `balanced_pixels_random` | 5.047769 | `score_unstable` |
| Pixel | 4 | `sg_smooth` | `balanced_pixels_center` | 2.797507 | `candidate` |
| Pixel | 5 | `raw` | `balanced_pixels_center` | 2.794307 | `candidate` |

The shortlist contains ten distinct preprocessing names and five rows per expected family.

### Detailed plotted candidates

The four detailed variants were:

- balanced random + `snv_sg_smooth` (`score_unstable`);
- object mean + `absorbance_sg_d1` (`candidate`);
- object median + `absorbance_sg_d1` (`candidate`);
- balanced center + `snv_sg_smooth` (`batch_sensitive`).

## Interpretation guidance

### Do not equate PCA separation with classification performance

PCA maximizes total variance without using class labels. Class diagnostics are calculated after fitting and can look favorable in-sample without guaranteeing robust SIMCA sensitivity or specificity.

### Treat batch effects as experimental evidence

A low batch trace ratio is desirable, but visual score patterns and acquisition design should also be checked. Batch labels can proxy instrument drift, sample history, position, or class imbalance.

### Inspect loadings with transformed units in mind

Raw, absorbance, SNV, MSC, smoothed, and derivative loadings have different meanings. A large derivative loading indicates sensitivity to local slope/curvature, not a large reflectance band contribution.

### Keep object-level decisions central

Pixel PCA is ranked using object-aggregated metrics because the final application detects peanut objects. Downstream pixel models still require an explicit pixel-to-object decision rule and grouped validation by `object_id`.

### Use warning fields, not rank alone

The recorded global top candidate and two saved pixel shortlist rows are flagged as unstable. Shortlist membership means “forward for downstream validation,” not “approved model.”

## Recommended review checklist

### Data and design

- Confirm that only pure almond and peanut objects from batches 1–3 are included.
- Verify batch and class counts and preserve batch 4 for external testing.
- Confirm the active wavelengths and any optional selected window.
- Check that raw pixel reflectance is physically suitable for absorbance conversion.

### PCA results

- Inspect explained variance and whether requested component counts are sufficient.
- Review class and batch score patterns together.
- Check object-aggregated results for pixel matrices.
- Examine PC1–PC3 loadings for plausible spectral structure.
- Review Q/T² extremes at the source object/image level.
- Confirm that rankings are not driven only by scale-sensitive metrics.

### Selection

- Audit contribution columns in `pca_scoring_diagnostics.parquet`.
- Inspect score and rank stability standard deviations.
- Read the full `pca_validation_warning`, not only the first `selection_flag`.
- Decide whether flagged candidates may proceed to SIMCA validation.
- Confirm exactly which variant supplied each shortlisted preprocessing.
- Do not interpret missing projection penalties as evidence of projection stability.

### Reproducibility

- Preserve the HDF5 database version or hash.
- Save the full PCA protocol and wavelength configuration if results are audited.
- Record source-code version and dependency versions.
- Export diagnostic figures required for review.
- Rebuild stateful preprocessing and models under leakage-safe validation in downstream notebooks.

## Common modifications

### Enable the wavelength window

```python
USE_WAVELENGTH_WINDOW = True
WINDOW_MIN_NM = 1225.0
WINDOW_MAX_NM = 1675.0
```

Verify the actual selected wavelengths and Savitzky–Golay compatibility after changing the feature count.

### Include all pixels

```python
RUN_ALL_PIXELS = True
```

This adds 18 more PCA combinations, increases memory/time, weights objects by area, and requires grouped downstream validation.

### Change shortlist size

Change the project-level configuration rather than only a local display limit:

```python
PCA_SELECTION_CONFIG = expcfg.make_pca_selection_config(
    max_preprocessings_per_family=3,
)
```

### Require warning-free shortlist rows

Filter the candidate pool before selecting, after deciding how to handle families with too few passing rows:

```python
pca_scored_pass_df = pca_scored_df[
    pca_scored_df["pca_validation_pass"]
].copy()
```

### Evaluate held-out batch 4

Use batch 4 only after freezing the selection policy. Fit preprocessing and PCA on batches 1–3, transform batch 4 with the fitted objects, and calculate projection Q/T² and centroid-shift metrics without refitting.

### Save the PCA protocol

```python
save_parquet(
    pca_protocol_df,
    RESULTS_DIR / "pca_protocol.parquet",
)
```

## Troubleshooting

### `RuntimeError: Could not find project root`

Launch Jupyter from the project root or its direct `notebooks/` directory and confirm that `src/` exists.

### Database loading fails

Verify `DB_H5_PATH`, rebuild the HDF5 database if needed, and confirm it with Notebook 01. Notebook 03 has no explicit file-existence preflight.

### `PCA_ALLOWED_BATCHES must stay [1, 2, 3]`

The notebook deliberately enforces the project protocol. Change the protocol and downstream assumptions together rather than bypassing the assertion casually.

### No object is selected

Inspect stored `sample_kind`, `object_nut_type`, and numeric batch values. Confirm that the expected classes and batches exist.

### PCA eigendecomposition fails

Check for NaN or infinite values after preprocessing, constant/degenerate features, and adequate observation count. The run-level loop re-raises exceptions and does not save partial failure diagnostics.

### `ncomp_95` exceeds `N_COMPONENTS`

The variance threshold uses the full covariance eigenvalue spectrum, while loadings/scores retain only `N_COMPONENTS`. Increase `N_COMPONENTS` and rerun if later steps require that threshold.

### Scores or loadings plots cannot find metadata

Check row-aligned metadata from `build_matrix`. The helper supports current and legacy key names, but missing object, source-image, or batch metadata reduces diagnostic detail.

### Class-difference calculation fails without wavelengths

Modify Cell 37 to return the band index when `wavelengths is None` rather than indexing a missing array.

### Shortlist validation fails after saving

Inspect family names, row counts, Parquet serialization, and required selection fields. The validator requires both expected families and no more than the configured maximum per family.

### Memory use is high

The registry retains transformed matrices, PCA objects, preprocessors, scores, and loadings for all 72 combinations. Balanced matrices have 12,360 rows. Disable detailed plots, avoid heavy-array reconstruction, reduce combinations, or persist/reload only selected results.

## Known limitations and maintenance notes

- Notebook 03 does not enforce Notebook 01 QC or reuse Notebook 02 result contracts.
- There is no friendly input preflight, overwrite guard, database hash, code version, or timestamp.
- Heavy arrays are reconstructed although PCA does not require most of them.
- Wavelength configuration and detailed wavelength selection are displayed but not saved.
- `WAVELENGTH_MODE` remains the default label under custom windowing.
- The `warnings` import and the NumPy generator in Cell 8 are unused.
- The batch subset is rigidly asserted to `[1, 2, 3]`.
- All PCA metrics are calculated in-sample; no cross-validation is performed.
- Held-out batch 4 is excluded but not projected or evaluated.
- Projection-weighted score terms are configured but inactive because no projection data are passed.
- PCA centers features but does not autoscale bands.
- `ncomp_90` and `ncomp_95` can exceed the retained score dimension.
- Q/T² summaries and diagnostics use three components, not all 20 retained components.
- Balanced random sampling inherits the per-object seed-reset issue from `redim_matrix.py`.
- Absorbance preprocessing clips nonpositive pixel reflectance rather than flagging it.
- Effective `sg_d2` polynomial order differs from the saved protocol value.
- A single failing preprocessing/matrix run aborts execution; partial failures are not tabulated.
- Backward-compatibility column synthesis can hide workflow-version differences.
- Global scores are relative, variant-scaled values and are not absolute model-quality measures.
- Score bootstrap stability does not refit PCA or resample observations.
- Relative flags depend on the evaluated candidate pool and quantiles.
- `selection_flag` reports only the first warning; use `pca_validation_warning` for the complete list.
- The shortlist does not exclude warned candidates.
- Shortlist validation checks family/size contracts, not scientific acceptability.
- Object/pixel ranking separation uses name prefixes instead of `matrix_family`.
- The class-difference table compares quantities with different transformed units and fails without wavelengths.
- Fitted models, preprocessors, scores, loadings, transformed matrices, protocol, and figures are not persisted.

## Downstream use

After reviewing the scored summary, diagnostics, warnings, loadings, score patterns, and shortlist, continue with:

```text
04A_simca_grid_validation.ipynb
```

Notebook 04A should treat `pca_selected_preprocessings.parquet` as a bounded candidate set, not a final decision. It should rebuild matrices, fit preprocessing only on training data, group validation by object, evaluate held-out data, and preserve warning/provenance information for every candidate.
