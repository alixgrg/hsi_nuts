# 02_matrices_preprocessing.ipynb

## Purpose

This notebook prepares and validates the matrix representations and spectral preprocessing methods used by PCA and SIMCA. It loads the canonical object database, optionally applies a wavelength window, builds object-level and pixel-level matrices, applies preprocessing chains, and saves summary tables.

Notebook 02 is the bridge between the database and modeling notebooks. It does not choose final models; it documents whether candidate matrices and preprocessing chains are technically usable.

## Main Inputs

- `HSI Data/processed/nir_uco_database.h5`
- Shared experiment configuration: `src/experiment_config.py`
- Matrix methods listed in `MATRIX_METHODS_TO_CHECK`
- Preprocessing chains listed in `PREPROCESSING_CONFIGS_TO_COMPARE`

## Main Outputs

- `results/02_matrices_<RESULTS_TAG>/wavelength_config.parquet`
- `results/02_matrices_<RESULTS_TAG>/matrix_summary.parquet`
- `results/02_matrices_<RESULTS_TAG>/preprocessing_summary.parquet`
- `results/02_matrices_<RESULTS_TAG>/matrix_preprocessing_errors.parquet` when failures occur
- Diagnostic plots for spectra, matrix dimensions, and metadata coverage

## Execution Logic

1. Detect project root and import local modules.
2. Load the canonical HDF5 database.
3. Optionally restrict the spectral axis with a wavelength window.
4. Define pure-sample filters and reference-class filters.
5. Build matrix variants from object records:
   - `object_mean`
   - `object_median`
   - `balanced_pixels`
   - `all_pixels`
6. Apply spectral preprocessing chains using `SpectralPreprocessor`.
7. Summarize matrix shapes, labels, metadata, and preprocessing outputs with `src.workflows.matrix_preprocessing`.
8. Save wavelength, matrix, preprocessing, and error summaries.
9. Display diagnostic plots for matrix and preprocessing sanity checks.

## How To Use

1. Run notebooks 00 and 01 first.
2. Keep `WAVELENGTH_MODE` and `RESULTS_TAG` initialized from `expcfg`.
3. Leave `USE_WAVELENGTH_WINDOW=False` for the canonical `non_noisy_all` workflow.
4. Use `PURE_REFERENCE_FILTERS` for reference-class matrix checks.
5. Use `PURE_BATCH_12_FILTERS` to inspect the training subset used later by SIMCA.
6. Keep `MATRIX_METHODS_TO_CHECK` aligned with downstream notebooks when adding a matrix family.
7. Review `matrix_preprocessing_errors.parquet` before moving to notebook 03.

## Key Parameters

- `MATRIX_METHODS_TO_CHECK`: matrix families/variants evaluated in this notebook.
- `M_BALANCED_PIXELS`: number of pixels sampled per object for balanced pixel matrices.
- `BALANCED_PIXEL_STRATEGIES`: sampling strategies for balanced pixel matrices.
- `PREPROCESSING_CONFIGS_TO_COMPARE`: compact list of preprocessing chains used for visual and numerical checks.
- `SG_WINDOW_LENGTH` and `SG_POLYORDER`: Savitzky-Golay parameters passed to `SpectralPreprocessor`.
- `REFERENCE_CLASSES`: initialized from `src/experiment_config.py`.
- `MATRIX_SUMMARY_REQUIRED_COLUMNS`: required schema for `matrix_summary.parquet`.
- `PREPROCESSING_SUMMARY_REQUIRED_COLUMNS`: required schema for `preprocessing_summary.parquet`.

## Associated Modules And Functions

### `src.io.database_h5`

- `load_nir_uco_h5(path, reconstruct_heavy_object_arrays=True)`: loads object and image records for matrix construction.

### `src.matrices.matrix_registry`

- `MatrixOutput`: formal matrix-construction contract with `X`, `y`, `metadata`, `wavelengths`, `matrix_method`, and `matrix_spec`.
- `build_matrix_output(object_db, matrix_method, filters, ...)`: returns a validated `MatrixOutput`.
- `build_matrix(object_db, matrix_method, filters, ...)`: converts filtered object records into a modeling matrix plus labels and metadata. It remains backward-compatible with existing notebooks and can return `X, y, metadata, wavelengths` with `return_wavelengths=True`.
- `available_matrix_methods()`: returns matrix methods registered in the project.
- `get_matrix_spec(matrix_method)`: returns metadata about one registered matrix method.
- `MatrixSpec`: describes a matrix method, its family, and construction behavior.

### `src.spectra.preprocessing`

- `SpectralPreprocessor`: fit/transform class for spectral preprocessing chains.
- `reflectance_to_absorbance(...)`: converts reflectance spectra to absorbance.
- `snv(...)`: applies standard normal variate normalization.
- `msc_fit(...)` and `msc_transform(...)`: fit and apply multiplicative scatter correction.
- `savgol_derivative(...)`: applies Savitzky-Golay smoothing or derivatives.

### `src.spectra.preprocessing_configs`

- `DEFAULT_PREPROCESSING_CONFIGS`: broad default preprocessing dictionary.
- `SIMCA_SEARCH_PREPROCESSING_CONFIGS`: candidate preprocessing dictionary for SIMCA search.
- `normalize_preprocessing_configs(...)`: converts aliases or explicit step lists into validated preprocessing chains.

### `src.spectra.band_selection`

- `select_wavelength_range_from_database(...)`: slices image and object spectral arrays to a wavelength interval.
- `wavelength_selection_summary(info)`: formats wavelength-window metadata as a table.

### `src.workflows.matrix_preprocessing`

- `summarize_matrix_output(...)`: creates one `matrix_summary.parquet` row and aligned metadata dataframe for a built matrix.
- `summarize_preprocessing_output(...)`: creates one `preprocessing_summary.parquet` row for a transformed matrix.
- `validate_required_columns(...)`: blocks execution if a result table misses its required column contract.

### `src.visualization`

- `plot_spectra(...)`: displays representative spectra after preprocessing.
- `plot_spectra_by_batch(...)`: checks batch effects in spectra.
- `plot_bar_values(...)`: displays scalar summaries.
- `plot_counts_by_group(...)`: displays matrix coverage by class, batch, or sample kind.

### `src.utils`

- `save_parquet(...)`: saves required result tables.
- `save_parquet_if_nonempty(...)`: saves error tables only when failures occur.

### `src.experiment_config`

- Provides shared values for wavelength mode, target class, reference classes, random state, balanced-pixel sampling, train-batch filters, and notebook 02 result-table column contracts.

## Maintenance Checks

- Each matrix method should produce expected labels and metadata columns.
- Matrix construction should validate the row contract: `X.shape[0] == len(y) == len(metadata[col])`.
- When available, `wavelengths` must have the same length as `X.shape[1]`.
- Balanced-pixel matrices should preserve object identifiers so pixel-level analyses can be traced back to objects.
- Preprocessing summaries should report finite values and should not silently drop classes or batches.
- Errors should be saved in `matrix_preprocessing_errors.parquet`, not hidden by broad exception handling.
- If `SpectralPreprocessor` changes, rerun this notebook and notebook 03.

## Automated Tests

- `tests/test_notebook02_matrices_preprocessing.py` covers `build_matrix()` and `build_matrix_output()` for object, pixel, and balanced-pixel matrices.
- The tests verify dynamic filters on records, dataframes, and matrix construction.
- The tests cover stable preprocessing names, `normalize_preprocessing_configs()`, `SpectralPreprocessor`, and the column contracts for `matrix_summary.parquet` and `preprocessing_summary.parquet`.
