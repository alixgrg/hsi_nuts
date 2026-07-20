# 01_database_quality_check.ipynb

## Purpose

This notebook validates the canonical HDF5 database created by notebook 00. It checks image-level metadata, object extraction quality, segmentation behavior, spectral distributions, and optional raw-versus-processed consistency.

The notebook is diagnostic: it does not build new model inputs. Its role is to detect database issues before matrix construction and PCA exploration.

## Main Inputs

- `HSI Data/processed/nir_uco_database.h5`
- Optional raw MATLAB file for raw database comparison
- Shared experiment configuration: `src/experiment_config.py`
- QC flags and plotting parameters from the notebook parameter cell

## Main Outputs

- `results/01_quality_check/image_qc_summary.parquet`
- `results/01_quality_check/object_qc_summary.parquet`
- `results/01_quality_check/qc_flags.parquet` when QC flags are generated
- Interactive or static QC plots displayed in the notebook

## Execution Logic

1. Detect the project root and import local modules.
2. Validate that the HDF5 database from notebook 00 exists.
3. Load image and object databases with heavy object arrays reconstructed.
4. Build image and object QC summary tables with `src.workflows.quality_check`.
5. Display representative image and segmentation overlays.
6. Display object grids and object area distributions.
7. Plot spectral examples and distributions by class or metadata group.
8. Optionally compare raw MATLAB content against the processed database.
9. Check missing fields and object shape consistency.
10. Save QC summaries and non-empty QC flag tables.

## How To Use

1. Run notebook 00 first.
2. Keep `RECONSTRUCT_HEAVY_OBJECT_ARRAYS=True` for visual and spectral checks.
3. Use the `RUN_*` flags to control expensive or verbose QC sections.
4. Increase `N_EXAMPLE_IMAGES_PER_KIND` or `N_OBJECTS_IN_GRID` only when investigating a specific issue.
5. Review the QC output before running notebook 02.

## Key Parameters

- `RUN_SEGMENTATION_QC_PLOTS`: controls segmentation overlay diagnostics.
- `RUN_OBJECT_QC_PLOTS`: controls object-grid and object-area diagnostics.
- `RUN_SPECTRAL_QC_PLOTS`: controls spectral distribution diagnostics.
- `RUN_RAW_DB_COMPARISON`: enables optional comparison with the raw MATLAB file.
- `RANDOM_STATE`: initialized from `src/experiment_config.py` for reproducible sampling.
- `MAX_OBJECT_SPECTRA_PER_GROUP`: limits plotted spectra per group.

## Associated Modules And Functions

### `src.io.database_h5`

- `load_nir_uco_h5(path, reconstruct_heavy_object_arrays=True)`: reads the canonical database created by notebook 00.

### `src.io.dataload`

- `load_mat_file(path)`: loads the raw MATLAB file when raw-versus-processed checks are enabled.

### `src.data.database`

- `preprocess_nir_uco_cube(...)`: applies the same low-level cube preprocessing used in notebook 00, useful for raw comparison checks.

### `src.workflows.quality_check`

- `build_image_qc_table(image_db)`: builds `image_qc_summary.parquet` rows from image records.
- `build_image_qc_warnings(image_qc_df)`: creates image-level warning rows.
- `build_object_qc_table(object_db)`: builds `object_qc_summary.parquet` rows from object records.
- `build_object_qc_warnings(object_qc_df)`: creates object-level warning rows.
- `check_missing_required_fields(image_db, object_db)`: reports missing required image/object fields.
- `build_object_shape_check_tables(object_db, image_db)`: checks object spectra, positions, summary spectra, and image-band consistency.
- `build_qc_flags_table(...)`: combines all warning sources into the optional `qc_flags.parquet` schema.

### `src.visualization.plot_images`

- `plot_image2d(...)`: displays a single 2D image or band-derived image.
- `plot_label_overlay_from_image_db(...)`: overlays object labels on processed image records.

### `src.visualization.plot_spectra`

- `plot_spectra(...)`: displays selected spectra or summary curves.
- `plot_spectral_distribution(...)`: compares spectral distributions across groups.

### `src.visualization.plot_generic`

- `plot_bar_values(...)`: visualizes scalar QC summaries.
- `plot_counts_by_group(...)`: displays counts by metadata groups such as class, batch, or sample kind.

### `src.visualization.plot_objects`

- `plot_object_view(...)`: inspects one extracted object.
- `plot_object_grid(...)`: inspects multiple extracted objects.
- `plot_object_area_distribution(...)`: checks object-size distributions.

### `src.utils`

- `save_parquet(df, path, ...)`: saves QC tables.
- `save_parquet_if_nonempty(df, path, ...)`: writes optional QC flags only when rows exist.

### `src.experiment_config`

- Provides `RANDOM_STATE` so all sampled QC displays are reproducible.

## Maintenance Checks

- The notebook should raise immediately if the HDF5 database is missing.
- QC summaries should contain all expected batches and sample kinds.
- Object area distributions should not show obvious segmentation failure modes.
- Spectral plots should not show unexplained class-wide artifacts or discontinuities.
- Any QC issue that affects matrix construction should be fixed in notebook 00, then notebooks 01 to 03 should be rerun.

## Automated Tests

- `tests/test_notebook01_quality_check.py` builds image/object QC tables on a mini fixture.
- The tests verify that an empty QC flag table keeps the expected columns and does not require writing `qc_flags.parquet`.
- A non-empty warning fixture verifies the canonical flag columns: `record_type`, `record_id`, `flag_type`, `warning`.
