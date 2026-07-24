# Notebook 02 — Modelling Matrices and Spectral Preprocessing

Documentation for `02_matrices_preprocessing.ipynb`.

## Purpose

This notebook documents and validates the conversion of the NIR UCO object database into numerical matrices suitable for PCA, SIMCA, and later anomaly-detection workflows.

It compares four modelling representations:

- one mean spectrum per object;
- one median spectrum per object;
- a controlled number of sampled pixels per object;
- every segmented object pixel.

It then fits and visualizes 17 spectral preprocessing configurations, compares batch behavior, and smoke-tests every configured matrix/preprocessing combination.

The notebook saves compact configuration and summary tables for downstream notebooks. It does **not** save the complete matrices, fitted preprocessors, or figures.

> **Scope:** this notebook is a matrix-construction and preprocessing validation stage. It does not fit PCA or SIMCA models, select a final preprocessing method, optimize hyperparameters, or evaluate anomaly-detection performance.

> **Important:** compatibility means that a transformation completed without raising an exception. It is not evidence that the representation is statistically appropriate, free from leakage, physically meaningful, or optimal for classification.

## Position in the workflow

```text
00_building_database.ipynb
    -> nir_uco_database.h5
01_database_quality_check.ipynb
    -> structural and visual QC
02_matrices_preprocessing.ipynb
    -> matrix_summary.parquet
    -> preprocessing_summary.parquet
03_pca_exploration_selection.ipynb
    -> representation and preprocessing selection
```

Notebook 02 loads the HDF5 database directly. It does not read the Parquet QC outputs from Notebook 01 and does not enforce a previous QC acceptance decision. The user must therefore confirm that the database has passed the intended quality review before running this notebook.

## Processing overview

The notebook performs the following operations:

1. Locate the project root and import the project modules.
2. Configure the database path, wavelength mode, matrix methods, sampling, and preprocessing chains.
3. Load the image and object databases from HDF5.
4. Optionally restrict all spectral arrays to a wavelength window while preserving object geometry and segmentation.
5. Inspect object metadata and the active wavelength axis.
6. Query the matrix registry.
7. Build five matrix variants from pure almond and peanut objects.
8. Validate row/label/metadata/wavelength alignment and save matrix summaries.
9. Compare random and center-based balanced-pixel sampling.
10. Visualize raw object-mean spectra.
11. Normalize and fit 17 preprocessing chains on the object-mean matrix.
12. Save preprocessing summaries and inspect class- and batch-level curves.
13. Compare Savitzky–Golay smoothing, first derivatives, and second derivatives.
14. Apply every preprocessing chain to every matrix variant in a compatibility smoke test.
15. Display the broader default and focused SIMCA preprocessing registries.

## Expected project layout

```text
project_root/
├── notebooks/
│   ├── 00_building_database.ipynb
│   ├── 01_database_quality_check.ipynb
│   └── 02_matrices_preprocessing.ipynb
├── src/
│   ├── experiment_config.py
│   ├── utils.py
│   ├── io/
│   │   └── database_h5.py
│   ├── matrices/
│   │   ├── matrix_registry.py
│   │   └── redim_matrix.py
│   ├── spectra/
│   │   ├── band_selection.py
│   │   ├── preprocessing.py
│   │   └── preprocessing_configs.py
│   ├── workflows/
│   │   └── matrix_preprocessing.py
│   └── visualization/
│       ├── plot_generic.py
│       └── plot_spectra.py
├── HSI Data/
│   └── processed/
│       └── nir_uco_database.h5
└── results/
```

The notebook creates a wavelength-tagged result directory such as `results/02_matrices_non_noisy_all/`.

## Requirements

The recorded notebook kernel is named `hsi-nuts`. Use the same project environment as the preceding notebooks. The relevant packages include:

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

Prefer a pinned project environment so that numerical, sampling, Savitzky–Golay, and serialization behavior remains reproducible.

## How to run the notebook

1. Build and quality-check the HDF5 database with Notebooks 00 and 01.

2. Verify that this file exists:

   ```text
   HSI Data/processed/nir_uco_database.h5
   ```

3. Activate the project environment.

4. Start Jupyter from the project root or its direct `notebooks/` directory:

   ```bash
   jupyter lab notebooks/02_matrices_preprocessing.ipynb
   ```

5. Review the wavelength mode, matrix filters, balanced-pixel settings, Savitzky–Golay parameters, and preprocessing registry.

6. Restart the kernel and run every cell in order.

7. Inspect matrix sizes, class row counts, non-finite rates, spectral plots, batch facets, and compatibility results.

8. Continue to `03_pca_exploration_selection.ipynb` only after the matrix semantics and preprocessing behavior are acceptable.

The notebook uses `%autoreload 2`. Restart and rerun all cells after editing a source module to obtain a reproducible final result.

## Configuration reference

### Input and wavelength settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `DB_H5_PATH` | `HSI Data/processed/nir_uco_database.h5` | Database produced by Notebook 00. |
| `USE_WAVELENGTH_WINDOW` | `False` | Keep all stored non-noisy bands when false; slice databases to a wavelength interval when true. |
| `WAVELENGTH_MODE` | `expcfg.DEFAULT_WAVELENGTH_MODE` | Descriptive project label, currently `non_noisy_all`. |
| `WINDOW_MIN_NM` | `1225.0` | Requested lower bound for optional wavelength selection. |
| `WINDOW_MAX_NM` | `1675.0` | Requested upper bound for optional wavelength selection. |
| `RESULTS_TAG` | `non_noisy_all` or window bounds | Controls the result-directory suffix. |

With the current database and `USE_WAVELENGTH_WINDOW=False`, 63 bands from approximately 960.74 to 1702.00 nm are used.

If the configured inclusive 1225–1675 nm window is enabled on the recorded wavelength axis, it selects the available bands lying inside the requested interval, not interpolated boundary values. This corresponds to 37 bands from approximately 1235.72 to 1666.13 nm.

### Matrix construction settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `MATRIX_METHODS_TO_CHECK` | four named methods | Object mean, object median, balanced pixels, and all pixels. |
| `M_BALANCED_PIXELS` | `40` | Requested rows per object for a balanced-pixel matrix. |
| `REPLACE_BALANCED_PIXELS` | `False` | Do not duplicate pixels when an object contains fewer than 40 pixels. |
| `BALANCED_PIXEL_STRATEGIES` | `random`, `center` | Sampling strategies compared for balanced pixels. |
| `RANDOM_STATE` | `42` | Seed used by matrix sampling and spectral-plot sampling. |
| `TARGET_CLASS` | `peanut` | Project target class imported from experiment configuration. |
| `REFERENCE_CLASSES` | `almond`, `peanut` | Classes included in matrix construction. |

### Filters

The matrix-building loop uses only:

```python
PURE_REFERENCE_FILTERS = {
    "sample_kind": ["pure"],
    "object_nut_type": ["almond", "peanut"],
}
```

Therefore:

- mixture objects are excluded because their object class is unknown;
- position-reference objects are excluded;
- all pure almond and peanut batches are included;
- the stored `split` field is ignored.

The preceding database assigned every object to `projection`, but that does not prevent pure objects from entering these exploratory matrices because no split filter is applied.

`PURE_FILTERS` and `PURE_BATCH_12_FILTERS` are defined in the parameter cell but are not used by the current notebook. `TARGET_CLASS` is also assigned locally but is not referenced later.

### Preprocessing settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `SG_WINDOW_LENGTH` | `11` | Savitzky–Golay window requested for smoothing and derivatives. |
| `SG_POLYORDER` | `2` | Polynomial order requested by the notebook. |
| `MAX_SPECTRA_TO_PLOT` | `80` | Maximum individual spectra shown in sampled line plots. |
| `PREPROCESSING_CONFIGS_TO_COMPARE` | 17 chains | Methods fitted, summarized, plotted, and compatibility-tested. |

The actual second-derivative implementation automatically raises the polynomial order to at least 3. Consequently, `sg_d2` uses an effective `(window_length=11, polyorder=3, deriv=2)` configuration even though `preprocessing_summary.parquet` records the notebook-level `sg_polyorder` value `2` for every row.

## Matrix representations

Every matrix builder returns:

```text
X        two-dimensional float array, shape (observations, spectral bands)
y        one label per observation
metadata one aligned metadata value per observation and field
```

The formal `MatrixOutput.validate()` contract checks that:

- `X` is two-dimensional;
- `len(y) == X.shape[0]`;
- every metadata array has one value per row;
- the wavelength-axis length matches `X.shape[1]` when wavelengths exist.

It also verifies that all selected objects with stored wavelength axes use consistent values.

### `object_mean`

One row is created for every selected object using `obj["mean_spectrum"]`.

```text
row weight: one per object
label: object_nut_type
metadata fields: 8
```

Every object contributes equally regardless of segmented area. This is often the clearest object-level representation for PCA and object-level decisions.

### `object_median`

Identical row semantics to `object_mean`, but uses `obj["median_spectrum"]`. The median can be less sensitive to unusual pixels within an object but discards within-object distribution information.

### `balanced_pixels`

Selects up to `m` rows from each object's pixel-level `spectra` matrix and stores global row/column coordinates in metadata.

With `replace=False`:

```text
rows contributed by object k = min(m, number of pixels in object k)
```

This reduces area-based weighting but does not guarantee exactly the same number of rows for objects smaller than `m`. It also does not balance classes: a class with more objects still contributes more rows.

With `replace=True`, objects smaller than `m` are padded by repeated pixel selections so every object contributes exactly `m` rows.

#### Random strategy

- When an object has at least `m` pixels, select `m` without replacement.
- When it has fewer than `m` pixels and replacement is disabled, keep all its pixels.
- When replacement is enabled, sample `m` pixels with replacement.

The current implementation reinitializes a random generator with the same `random_state` inside the per-object selection function. Objects with the same number of pixels can therefore receive identical relative index patterns. An outer generator is created in `_objects_to_balanced_pixel_matrix` but is not used. For statistically independent deterministic samples, pass a shared generator or derive an object-specific seed.

#### Center strategy

Computes the Euclidean distance between each global pixel position and the stored object centroid, sorts pixels by increasing distance, and selects the closest `m`. The alias `center_closest` has the same implementation.

This strategy reduces border-pixel exposure but can systematically emphasize object cores and change the sampled spectral distribution. It is a modelling choice, not merely an acceleration technique.

### `all_pixels`

Concatenates every selected object's pixel-level spectra. Each pixel is one observation.

Large objects contribute more rows and therefore more weight. Models or cross-validation procedures using this matrix must preserve object grouping; randomly splitting rows would place pixels from the same nut in both training and validation sets and cause leakage.

The registry also contains `pixel`, an alias for `all_pixels`. It is displayed in the registry table but is not built separately in this notebook.

### Row-level metadata

All representations include:

| Field | Meaning |
| --- | --- |
| `object_id` | Source object identifier. |
| `label` | `object_nut_type`, used as `y`. |
| `source_image` | Clean source-image key. |
| `source_clean_key` | Same clean source-image key. |
| `source_image_id` | Original source-image identifier, such as a key ending in `_sb`. |
| `batch` | Acquisition batch. |
| `area` | Full object area in pixels. |
| `sample_kind` | Pure, mixture, or position-reference status. |

Pixel representations add:

| Field | Meaning |
| --- | --- |
| `pixel_index` | Index within the object's stored spectral matrix. |
| `row`, `col` | Global image coordinates of the selected pixel. |

`source_image` is named like an image identifier but contains the clean key; `source_image_id` contains the original identifier.

## Optional wavelength selection

`select_wavelength_range_from_database` creates new top-level image and object dictionaries and restricts only spectral arrays:

- image `cube` and `wavelengths`;
- object `spectra`, `mean_spectrum`, `median_spectrum`, `std_spectrum`, `cube_crop`, and `wavelengths`;
- object and image band-count/range metadata.

Object IDs, masks, labels, positions, centroids, bounding boxes, and experimental metadata are preserved. The original `image_ref` is deliberately retained so segmentation is not changed when comparing wavelength windows. An optional `image_ref_selected_range` is added for visualization.

Object mean, median, and standard-deviation spectra are recomputed from the selected pixel spectra. The function returns modified copies rather than intentionally changing the original dictionaries in place; however, record copying is shallow, so unchanged nested objects and arrays may still be shared.

The reference wavelength axis is taken from the first usable image or object. The selection function does not first compare every record's original axis. Later matrix construction does check wavelength consistency among selected objects after the function has assigned the common selected axis.

When window selection is enabled, `wavelength_selection_df` containing requested bounds, actual bounds, selected indices, and selected wavelengths is displayed but not saved by the notebook. Only the more compact `wavelength_config.parquet` is persisted.

`WAVELENGTH_MODE` remains equal to the default project label even when a window is enabled. The `use_wavelength_window`, bounds, and `results_tag` fields must therefore be used together to interpret the configuration.

## Spectral preprocessing reference

Preprocessing steps are executed from left to right. Order matters because each transformation receives the output of the preceding step.

### `raw`

No transformation. Configuration validation requires `raw` to be used alone.

### `absorbance`

Converts reflectance to absorbance:

```text
A = log10(1 / R) = -log10(R)
```

Values below `eps` are clipped before the logarithm. `SpectralPreprocessor` passes its default `eps=1e-12`, so zero and negative reflectance values become an absorbance of 12 rather than producing NaN or infinity.

This guarantees finite numerical output for those values but can conceal nonphysical reflectance measurements and introduce extreme artificial values. The pure object-mean matrix is positive in the recorded run, whereas pixel-level matrices contain negative minima. Absorbance compatibility at pixel level should therefore be interpreted with care.

### `snv`

Standard Normal Variate normalization is applied independently to each spectrum:

```text
SNV(x) = (x - mean(x)) / sample_std(x)
```

The standard deviation uses `ddof=1`. Values below `eps` are replaced by 1 to avoid division by zero. NumPy's ordinary mean and standard deviation are used, so existing NaN values propagate.

### `msc`

Multiplicative Scatter Correction fits a reference spectrum equal to the column-wise mean of the matrix passed to `fit`. For each spectrum, least squares estimates an intercept and slope against the reference, then applies:

```text
x_corrected = (x - intercept) / slope
```

MSC is stateful. In downstream modelling, fit it on training data only and reuse the stored reference to transform validation, test, and mixture data. Fitting separately on all evaluation data would leak information and make transformations incomparable.

Notebook 02 uses the complete pure reference matrix for exploratory fitting and each complete matrix for the compatibility test. This is acceptable for a preprocessing smoke test, not for performance estimation.

### `sg_smooth`, `sg_d1`, and `sg_d2`

Use SciPy's Savitzky–Golay filter along the spectral axis with `mode="interp"`:

| Step | Derivative | Effective recorded-run parameters |
| --- | ---: | --- |
| `sg_smooth` | 0 | window 11, polynomial order 2 |
| `sg_d1` | 1 | window 11, polynomial order 2 |
| `sg_d2` | 2 | window 11, polynomial order 3 |

When wavelengths are available, derivative scaling uses the mean adjacent wavelength difference as `delta`; otherwise `delta=1`. This assumes sufficiently uniform spacing and does not model varying per-band intervals.

The window must be valid for the number of selected bands. Very narrow wavelength windows can make Savitzky–Golay configurations fail.

### `vector_norm`

Divides every spectrum by its L2 norm. It is a valid registered preprocessing step and alias, but it is not included in the notebook's 17 compared configurations or in `DEFAULT_PREPROCESSING_CONFIGS`.

### Compared chains

The notebook compares:

```text
raw
absorbance
snv
msc
sg_smooth
sg_d1
sg_d2
snv + sg_smooth
snv + sg_d1
snv + sg_d2
absorbance + snv
absorbance + sg_smooth
absorbance + sg_d1
absorbance + sg_d2
absorbance + snv + sg_smooth
absorbance + snv + sg_d1
absorbance + snv + sg_d2
```

`DEFAULT_PREPROCESSING_CONFIGS` additionally contains `absorbance_msc`, but that chain is only displayed in Cell 35 and is not fitted or compatibility-tested by this notebook.

## Detailed notebook walkthrough

Cell numbers are zero-based and correspond to positions in the `.ipynb` file.

### Cell 0 — Objectives

Introduces the four matrix representations, two balanced-pixel strategies, spectral preprocessing families, and summary outputs. It correctly states that PCA and SIMCA are outside this notebook.

### Cell 1 — Imports and project-root detection

Imports standard data tools, expands pandas display limits, locates `PROJECT_ROOT` by checking the current directory and its parent for `src/`, and inserts the root into `sys.path`.

### Cell 2 — Project imports and autoreload

Imports HDF5 loading, experiment configuration, matrix registry, preprocessor and configuration registries, band selection, plotting, Parquet helpers, and workflow summary functions. It enables `%autoreload 2` before importing workflow helpers.

### Cell 3 — Parameters and output paths

Defines wavelength handling, result tagging, matrix construction, filters, preprocessing chains, and plotting limits. The output directory is created immediately.

No overwrite guard is implemented. Re-running the notebook replaces the three mandatory Parquet summaries.

### Cell 4 — HDF5 loading

Loads `object_db` and `image_db` with heavy object-array reconstruction enabled and prints record counts and sample keys.

The notebook has no explicit `DB_H5_PATH.exists()` preflight. Missing or invalid files fail inside the loader.

Heavy reconstruction is not required by the matrix builders themselves: they need spectra, positions, centroid, wavelengths, and metadata, all available in compact storage. It increases memory use and is mainly relevant if extended code needs reconstructed crops or global masks.

### Cell 5 — Wavelength handling

Either creates wavelength-restricted database copies or reads the axis from the first object. It builds and saves a one-row wavelength configuration table. A detailed selection table is shown only when windowing is enabled.

This cell assumes `object_db` is nonempty because it calls `next(iter(object_db.values()))` when windowing is disabled.

### Cells 6–7 — Object inventory and wavelength display

Cell 6 flattens selected object metadata for inspection and groups counts by sample kind, object class, and batch. Cell 7 retrieves the wavelength axis from the first object again and prints its size and endpoints.

`object_meta_df` and its grouped inventory are not saved.

### Cells 8–9 — Matrix registry

Lists all registered methods and their formal levels, spectrum fields, descriptions, and pixel-sampling flags. The sorted registry includes the `pixel` alias in addition to the four main representations.

### Cell 10 — Matrix construction and validation

Builds five variants:

```text
object_mean
object_median
balanced_pixels | random
balanced_pixels | center
all_pixels
```

Each uses `PURE_REFERENCE_FILTERS`. `build_matrix` creates and validates a `MatrixOutput`, then returns `X`, `y`, and aligned metadata. `summarize_matrix_output` records dimensions, labels, unique object/image counts, NaN counts, and global value statistics.

The summary schema is checked against `expcfg.MATRIX_SUMMARY_REQUIRED_COLUMNS`. This verifies column presence, not scientific thresholds or expected values.

Full matrices and row-level metadata are kept in `matrix_examples` only for the current kernel session.

### Cell 11 — Save matrix summary

Writes `matrix_summary.parquet`.

### Cells 12–13 — Matrix size and metadata inspection

Plots the number of observations for every variant and prints each `X`, `y`, and metadata shape with sample rows.

The observation-count plot demonstrates computational size but should also be interpreted as a weighting comparison.

### Cells 14–16 — Balanced-pixel comparison

Concatenates metadata from random and center strategies, counts selected pixels per object, summarizes minimum/mean/maximum selected rows, and compares class row counts.

Both strategies select the same number of rows per object under the same replacement rule, so their count summaries match. Their selected pixel identities and spectral values differ.

The recorded class counts remain unequal—8,414 almond rows and 7,026 peanut rows per strategy—because balancing is performed by object rather than by class.

### Cells 17–18 — Raw object-mean spectra

Uses the object-mean matrix to display:

- up to 80 individual spectra sampled approximately equally by class;
- class mean curves with plus-or-minus-one-standard-deviation bands.

The uncertainty band describes between-object dispersion, not a confidence interval for the mean.

### Cells 19–20 — Preprocessing registry normalization

`normalize_preprocessing_configs` validates every chain, ensures that registered steps are known, and rejects `raw` inside a multi-step chain. It converts the configuration mapping into normalized tuples and a readable display table.

Mapping keys are treated as display names; normalization validates their associated steps but does not require the name itself to match those steps.

### Cell 21 — Fit preprocessing chains

For each of 17 configurations, creates a fresh `SpectralPreprocessor` and calls `fit_transform` on the complete object-mean matrix. It stores the steps, fitted preprocessor, and transformed matrix in memory.

`summarize_preprocessing_output` records shape, global distribution statistics, non-finite rate, and notebook-level Savitzky–Golay settings. The resulting table is checked against the required-column contract.

This loop has no per-method `try/except`. A failing preprocessing stops the notebook before the later compatibility table can be built.

### Cell 22 — Save preprocessing summary

Writes `preprocessing_summary.parquet`. Transformed matrices and fitted preprocessors are not serialized.

### Cell 23 — Class summaries for all preprocessing methods

Displays class mean and standard-deviation curves for all 17 transformed object-mean matrices. This creates 17 interactive figures and can make the notebook large.

### Cells 24–25 — Individual transformed spectra

Selects 13 named configurations and plots a stratified sample of at most 80 individual object spectra for each. Standalone MSC and standalone Savitzky–Golay methods are omitted from this detailed list even though their class summaries are shown in Cell 23.

### Cells 26–27 — Batch effects

Facets class mean and standard-deviation curves by batch for 15 candidate configurations. The `pure_mask` is redundant under the current pure-only matrix filter but protects the plotting code if broader matrices are used later.

The analysis is visual; no batch-effect statistic, hypothesis test, or automatic warning is calculated. Batch values are converted to strings for facet labels.

### Cells 28–30 — Savitzky–Golay comparisons

Filters the preprocessing summary to 12 smoothing/derivative variants and displays eight side-by-side figure pairs comparing smoothing with first or second derivatives under different upstream transformations.

The local loop variable is named `d1_name` even for pairs containing a second derivative; this is cosmetic and does not change execution.

### Cells 31–32 — Matrix/preprocessing compatibility

Applies all 17 configurations to all five matrix variants, producing 85 rows. Each combination receives a newly fitted preprocessor.

The loop catches exceptions and records status, error text, output shape, and non-finite rate. However, `status="ok"` is assigned whenever no exception occurs, even when `nan_rate > 0`. `errors_df` contains only exception failures.

For modelling, stateful methods such as MSC must not be independently fitted on validation or test matrices as this smoke test does.

### Cell 33 — Save compatibility failures

Writes `matrix_preprocessing_errors.parquet` only if `errors_df` is nonempty.

If a previous run produced an error file and the current run has no errors, the old file is not removed. Verify the final error count or use versioned/clean result directories.

Successful compatibility rows are displayed but not saved.

### Cells 34–35 — Downstream preprocessing sets

Displays the complete default configuration registry and the smaller SIMCA search registry. These tables are recommendations/registries only and are not saved here.

The default registry has 18 chains, including `absorbance_msc`; the SIMCA search registry has 12.

### Cell 36 — In-memory notebook configuration

Builds and displays a one-row DataFrame containing paths, wavelength settings, matrix methods, sampling parameters, filters, reference classes, Savitzky–Golay settings, and the 17 compared chains.

This configuration table is not persisted as a separate manifest.

### Cell 37 — Completion summary

Lists mandatory outputs, reports active bands, matrix variants, preprocessing methods, compatibility errors, and points to `03_pca_exploration_selection.ipynb`.

## Persisted outputs

| Output | Purpose |
| --- | --- |
| `wavelength_config.parquet` | Active wavelength mode, window flag/tag, requested bounds when active, band count, and actual wavelength endpoints. |
| `matrix_summary.parquet` | One row per matrix variant with dimensions, labels, provenance counts, and global numeric diagnostics. |
| `preprocessing_summary.parquet` | One row per compared preprocessing chain applied to the object-mean matrix. |
| `matrix_preprocessing_errors.parquet` | Optional exception-only compatibility failures. |

For the recorded run, the result directory is:

```text
results/02_matrices_non_noisy_all/
```

The notebook does not save:

- `X`, `y`, or row-level matrix metadata;
- fitted `SpectralPreprocessor` instances or MSC references;
- successful compatibility rows;
- detailed wavelength-selection indices;
- object inventory, balanced-sampling, or preprocessing-set tables;
- the notebook configuration manifest;
- Plotly figures.

Downstream notebooks must rebuild matrices and fit preprocessing from the persisted database and configuration choices.

## Output table schemas

### `wavelength_config.parquet`

| Column | Meaning |
| --- | --- |
| `wavelength_mode` | Descriptive project mode. |
| `use_wavelength_window` | Whether wavelength slicing was enabled. |
| `results_tag` | Directory/result identifier. |
| `window_min_nm`, `window_max_nm` | Requested bounds when windowing is active; otherwise missing. |
| `n_bands` | Active spectral feature count. |
| `min_wavelength_nm`, `max_wavelength_nm` | Actual active endpoints. |

### `matrix_summary.parquet`

| Column group | Columns |
| --- | --- |
| Identity | `matrix_method`, `balanced_pixel_strategy`, `filters` |
| Dimensions/classes | `n_observations`, `n_features`, `n_labels`, `labels` |
| Provenance | `has_metadata`, `n_unique_objects`, `n_unique_images` |
| NaN diagnostics | `n_nan_values`, `nan_rate` |
| Global values | `global_min`, `global_max`, `global_mean`, `global_std` |

`has_metadata` means that the metadata DataFrame and `y` have equal lengths. Matrix construction has already performed stronger per-field row-alignment validation before this summary is created.

### `preprocessing_summary.parquet`

| Column | Meaning |
| --- | --- |
| `preprocessing` | Readable configuration name. |
| `steps` | Ordered executable chain joined with ` + `. |
| `n_observations`, `n_features` | Shape of the transformed object-mean matrix. |
| `global_mean`, `global_std`, `global_min`, `global_max` | Whole-matrix descriptive statistics. |
| `nan_rate` | Fraction of values that are NaN or infinite. |
| `sg_window_length`, `sg_polyorder` | Notebook-level requested settings, not necessarily effective per-step settings for `sg_d2`. |

### `matrix_preprocessing_errors.parquet`

When present, contains:

```text
matrix_method, balanced_pixel_strategy, preprocessing, steps,
status, error, n_observations, n_features, nan_rate
```

It contains exception failures only. A transformation that returns non-finite values without raising an exception is marked `ok` and is not included.

## Reference run recorded in the notebook

### Database and wavelength configuration

| Metric | Recorded value |
| --- | ---: |
| Images loaded | 48 |
| Objects loaded | 1,262 |
| Pure reference objects selected | 394 |
| Pure almond objects | 214 |
| Pure peanut objects | 180 |
| Source images represented | 8 |
| Active bands | 63 |
| Wavelength range | approximately 960.74–1702.00 nm |
| Wavelength mode / tag | `non_noisy_all` |

### Matrix variants

| Matrix variant | Shape | Unique objects | Notes |
| --- | ---: | ---: | --- |
| `object_mean` | `(394, 63)` | 394 | One mean spectrum per object. |
| `object_median` | `(394, 63)` | 394 | One median spectrum per object. |
| `balanced_pixels`, random | `(15440, 63)` | 394 | Up to 40 pixels per object. |
| `balanced_pixels`, center | `(15440, 63)` | 394 | Up to 40 centroid-nearest pixels per object. |
| `all_pixels` | `(30197, 63)` | 394 | Every pixel from every pure reference object. |

Both balanced strategies selected between 15 and 40 pixels per object, with a mean of approximately 39.19. This confirms that some objects contain fewer than the requested 40 pixels and were not padded because replacement was disabled.

Per strategy, balanced rows were distributed as:

| Class | Rows |
| --- | ---: |
| Almond | 8,414 |
| Peanut | 7,026 |

All five matrix variants contained two labels, aligned metadata, zero NaN values, and 63 features.

Pixel-level raw reflectance included negative values in the recorded summaries:

- balanced random minimum: approximately `-0.1361`;
- balanced center minimum: approximately `-0.0322`;
- all-pixels minimum: approximately `-0.1379`.

These values should be reviewed before absorbance conversion because the current implementation clips them to `eps`.

### Preprocessing and compatibility

| Metric | Recorded value |
| --- | ---: |
| Preprocessing chains fitted on object means | 17 |
| Matrix variants | 5 |
| Compatibility combinations | 85 |
| Exception failures | 0 |
| Reported non-finite rate in every compatibility row | 0.0 |

The saved run therefore produced no `matrix_preprocessing_errors.parquet` file.

The very different scales of smoothed spectra and derivatives are expected. For example, the raw object-mean matrix had a global standard deviation near `0.104`, while standalone first and second derivatives had standard deviations near `8.06e-4` and `2.8e-5`. PCA and SIMCA configuration must account for the semantics and scale of each preprocessing rather than comparing raw numeric magnitudes directly.

## Interpretation and modelling implications

### Observation unit

The matrix method defines what one row means:

| Method | One row represents | Natural grouping unit |
| --- | --- | --- |
| Object mean/median | One nut | Source image or batch when required |
| Balanced pixels | One sampled pixel from a nut | Object ID |
| All pixels | One pixel from a nut | Object ID |

Metrics, cross-validation, and train/test splitting must use the same decision level intended for deployment. Pixel-level predictions often need an explicit aggregation rule to produce an object-level peanut decision.

### Weighting

- Object matrices weight every detected nut equally.
- Balanced-pixel matrices approximately weight every nut equally, except objects smaller than `m` when replacement is disabled.
- All-pixel matrices weight objects in proportion to area.
- None of these methods automatically equalizes almond and peanut class totals.

### Leakage prevention

For pixel matrices, group splits by `object_id`; otherwise pixels from the same physical nut can appear in both training and validation. Depending on the acquisition design, grouping by source image or batch may also be necessary.

Fit stateful preprocessing only on the training fold. In particular, the MSC reference must never be estimated from validation, test, mixture, or deployment spectra.

### Scientific selection

Zero exceptions and zero non-finite values do not determine which preprocessing is best. Selection should consider class separation, batch stability, robustness, interpretability, downstream cross-validation, and anomaly-detection assumptions.

## Recommended review checklist

### Wavelengths

- Confirm the active band count and actual endpoints.
- Confirm that all selected objects use identical wavelength axes.
- If windowing is enabled, record selected indices and actual available bounds.
- Verify that the window still contains enough bands for the SG window length.

### Matrices

- Confirm that labels, metadata, and rows align.
- Confirm that only intended sample kinds and classes pass the filters.
- Compare class, object, image, and batch counts.
- Inspect objects with fewer than `m` pixels.
- Decide whether replacement, center sampling, or all pixels matches the modelling objective.
- Preserve object IDs for grouped validation.

### Preprocessing

- Inspect raw negative or zero reflectance before absorbance conversion.
- Confirm that every transformed matrix is finite, even when no exception occurs.
- Check effective Savitzky–Golay parameters, not only requested summary values.
- Compare batches within each class.
- Fit stateful transformations on training data only in downstream experiments.
- Record the exact ordered chain used by every model.

### Persistence and reproducibility

- Keep the HDF5 database version or hash with summaries.
- Save the full notebook configuration if results must be audited.
- Remove or version stale optional error files.
- Rebuild matrices deterministically from the same code and random seed.
- Export figures required for review; notebook displays are not persisted automatically.

## Common modifications

### Enable the wavelength window

```python
USE_WAVELENGTH_WINDOW = True
WINDOW_MIN_NM = 1225.0
WINDOW_MAX_NM = 1675.0
```

After changing this option, verify the actual endpoints and feature count, then rerun all matrix and preprocessing compatibility checks.

### Enforce exactly 40 rows per object

```python
REPLACE_BALANCED_PIXELS = True
```

This duplicates pixels for small objects. Record duplicate-selection behavior because repeated spectra affect effective sample size.

### Use only SIMCA training batches

Replace the matrix filter with a batch-aware configuration, for example:

```python
PURE_REFERENCE_FILTERS = {
    "sample_kind": ["pure"],
    "object_nut_type": list(REFERENCE_CLASSES),
    "batch": list(expcfg.SIMCA_TRAIN_BATCHES),
}
```

This changes matrix sizes and must be reflected in result tags or manifests.

### Add vector normalization

```python
PREPROCESSING_CONFIGS_TO_COMPARE["vector_norm"] = ("vector_norm",)
```

### Add `absorbance_msc` to the tested set

```python
PREPROCESSING_CONFIGS_TO_COMPARE["absorbance_msc"] = (
    "absorbance",
    "msc",
)
```

### Treat non-finite output as a compatibility error

After transformation, explicitly fail or flag the combination:

```python
nan_rate = float(np.mean(~np.isfinite(Xp)))
if nan_rate > 0:
    raise ValueError(f"Non-finite output rate: {nan_rate:.3%}")
```

### Save the complete compatibility table

```python
save_parquet(
    compatibility_df,
    RESULTS_DIR / "matrix_preprocessing_compatibility.parquet",
)
```

## Troubleshooting

### `RuntimeError: Could not find project root`

Launch Jupyter from the project root or its direct `notebooks/` directory and confirm that `src/` exists.

### Database loading fails

Verify `DB_H5_PATH`, run Notebook 00, and confirm the HDF5 file with Notebook 01. Notebook 02 does not provide its own friendly existence check.

### `No objects found with the requested filters`

Inspect `object_meta_df`, `REFERENCE_CLASSES`, batch values, and sample-kind labels. Remember that filter values must match stored metadata exactly.

### Selected objects have inconsistent wavelength axes

Inspect each selected object's `wavelengths` values and rebuild the database if necessary. Do not suppress this check without establishing a shared physical feature axis.

### Savitzky–Golay filtering fails

Ensure the number of active bands is at least the effective window length, the window is a positive odd integer, and polynomial order is lower than the window length. Remember that `sg_d2` raises the polynomial order to at least 3.

### Absorbance output contains extreme values

Inspect zero and negative reflectance. Clipping to `eps` prevents numerical failure but can map invalid measurements to very large absorbance. Consider correcting, masking, or formally flagging those measurements before conversion.

### MSC results differ between datasets

Use the same fitted `SpectralPreprocessor` to transform every downstream subset. Independently fitting MSC changes the reference spectrum.

### Balanced matrices contain fewer than `m * n_objects` rows

This is expected with `replace=False` when objects contain fewer than `m` pixels. Enable replacement only if repeated pixels are acceptable.

### Random balanced samples appear correlated across objects

The current helper resets the generator to the same seed for every object. Refactor it to use one shared generator or deterministic object-specific seeds.

### No compatibility error file is produced

This means no exception was captured. Check `nan_rate` values separately. Also verify that a stale error file from an earlier run is not present.

### Memory use is high

`all_pixels`, 17 transformed versions, compatibility loops, full image cubes, and reconstructed heavy arrays can consume substantial memory. Disable heavy reconstruction, reduce checked matrices/configurations, or process combinations sequentially without retaining unnecessary arrays.

## Known limitations and maintenance notes

- Notebook 02 does not consume or enforce Notebook 01 QC results.
- There is no explicit input-file existence check or output overwrite guard.
- Heavy object arrays are reconstructed even though the current matrix workflow does not need most of them.
- `PURE_FILTERS`, `PURE_BATCH_12_FILTERS`, and the local `TARGET_CLASS` variable are unused.
- `pixel` duplicates `all_pixels` in the registry and is not separately tested.
- Balanced random sampling resets the same seed for every object; the outer random generator is unused.
- Balanced pixels approximately equalize objects, not classes, and small objects contribute fewer rows when replacement is disabled.
- All-pixel matrices weight objects by area and require grouped validation.
- Window selection starts from one reference axis and does not validate every original record before assigning the selected axis.
- `WAVELENGTH_MODE` remains the default label even when a custom window is active.
- Detailed wavelength-selection indices are not saved.
- `vector_norm` is supported but absent from the compared and default registries.
- `absorbance_msc` is displayed in the default registry but not tested by this notebook.
- Absorbance conversion clips nonpositive reflectance rather than flagging it.
- The preprocessing summary records polynomial order 2 for `sg_d2`, although the effective implementation uses at least 3.
- Cell 21 stops on the first preprocessing failure, before the per-combination error handling in Cell 32.
- Compatibility status ignores a positive non-finite rate unless the transform raises an exception.
- Stateful preprocessors are fitted independently on each complete matrix during the smoke test.
- Successful compatibility results, transformed matrices, fitted preprocessors, and most configuration tables are not persisted.
- A stale optional compatibility-error file can remain after a later successful run.
- The notebook generates many interactive figures but does not export them.
- No database hash, code version, timestamp, or complete saved manifest ties outputs to a specific run.

## Downstream use

After reviewing matrix semantics, spectral transformations, batch behavior, and compatibility, continue with:

```text
03_pca_exploration_selection.ipynb
```

The downstream modelling code should rebuild the selected matrix from `nir_uco_database.h5`, preserve row-level grouping metadata, fit stateful preprocessing on training data only, and record the chosen wavelength/matrix/preprocessing configuration with the resulting model.
