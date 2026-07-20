# 00_building_database.ipynb

## Purpose

This notebook builds the canonical NIR UCO HSI database used by the rest of the project. It reads the raw MATLAB file, parses image metadata, preprocesses the hyperspectral cubes, segments nut objects, extracts object-level records, and saves the database to HDF5 plus summary Parquet files.

Run this notebook first. Notebooks 01, 02, and 03 depend on the HDF5 file created here.

## Main Inputs

- Raw MATLAB file: `HSI Data/NIR camera UCO (889-1702 nm)/NIR_uco_sb.mat`
- Shared experiment configuration: `src/experiment_config.py`
- Segmentation parameters in the notebook parameter cell
- Optional `SELECTED_KEYS` list if only a subset of raw images should be processed

## Main Outputs

- `HSI Data/processed/nir_uco_database.h5`
- `results/00_database/image_summary.parquet`
- `results/00_database/object_summary.parquet`
- `results/00_database/database_manifest.parquet`
- Optional QC plots displayed in the notebook

## Execution Logic

1. Detect the project root and add it to `sys.path`.
2. Load project modules and the shared experiment configuration.
3. Define input/output paths and database construction parameters.
4. Validate that the raw `.mat` file exists and that overwriting is allowed.
5. Load raw MATLAB content with `load_mat_file()`.
6. Detect valid hyperspectral image keys with `detect_known_image_keys()`.
7. Resolve the final set of images to process with `resolve_selected_keys()`.
8. Build wavelengths with `make_wavelengths()`.
9. Build the minimal object database with `build_minimal_nir_uco_object_database()`.
10. Save the canonical image and object databases with `save_nir_uco_h5()`.
11. Reload the HDF5 file with `load_nir_uco_h5()` as a write/read integrity check.
12. Save inventory and manifest tables as Parquet.
13. Display representative segmentation and object QC plots.

## How To Use

1. Confirm that `RAW_MAT_PATH` points to the raw MATLAB file.
2. Keep `WAVELENGTH_MODE` and `RESULTS_TAG` initialized from `expcfg` unless the whole workflow is intentionally forked.
3. Adjust `SEGMENTATION_KWARGS` only when validating a segmentation change. `SEGMENTATION_KWARGS["min_area"]` is also the default object-extraction area threshold used by `build_minimal_nir_uco_object_database()`.
4. Set `SELECTED_KEYS` to a short list for debugging, or leave it as `None` for the full build.
5. Keep `OVERWRITE_OUTPUTS=True` when rebuilding the canonical database, otherwise set it to `False` to protect existing outputs.
6. Run all cells from top to bottom.

## Key Parameters

- `WAVELENGTH_MODE`: shared spectral namespace, currently `non_noisy_all`.
- `N_REMOVE_START`: number of noisy initial bands removed from the raw cube.
- `DATA_MODE`: stored data representation, currently reflectance.
- `OBJECT_MIN_AREA`: minimum area used both for segmentation cleanup and, by default, object extraction through `SEGMENTATION_KWARGS["min_area"]`.
- `FORCED_SPLIT`: split label assigned at database-build time.
- `INCLUDE_HEAVY_OBJECT_ARRAYS`: whether heavy per-object arrays are stored directly in the HDF5 object table.

## Associated Modules And Functions

### `src.io.dataload`

- `load_mat_file(path)`: loads the MATLAB file into a Python mapping. This is the raw entry point for the database build.

### `src.data.database`

- `parse_image_key(key, config=None)`: parses image names into structured metadata such as nut type, batch, sample kind, and mixture/reference information.
- `infer_split_from_metadata(meta)`: returns the default workflow split from parsed image metadata.
- `infer_object_nut_type_from_metadata(meta)`: returns the object label inferred from image metadata.
- `segmentation_metadata(mask, labels, threshold=None)`: builds stable image-level segmentation metadata used by the database records.
- `preprocess_nir_uco_cube(...)`: standardizes one raw hyperspectral cube before downstream object extraction.
- `build_minimal_nir_uco_object_database(...)`: orchestrates image preprocessing, segmentation, object extraction, metadata enrichment, and database assembly. If `min_area` is not passed explicitly, object extraction reuses `segmentation_kwargs["min_area"]`; pass `min_area` only when the extraction threshold intentionally differs from the segmentation cleanup threshold.
- `detect_known_image_keys(data, skip_non_cubes=True)`: filters raw MATLAB entries to valid hyperspectral cubes with known naming patterns.
- `resolve_selected_keys(data, selected_keys)`: returns the exact image keys to process, either all detected keys or a requested subset.

### `src.io.database_h5`

- `save_nir_uco_h5(...)`: writes image and object databases to the canonical HDF5 file.
- `load_nir_uco_h5(path, reconstruct_heavy_object_arrays=True)`: reloads the HDF5 database and optionally reconstructs heavy object arrays from image-level storage.
- `validate_nir_uco_h5(path)`: validates the HDF5 format marker, required groups, and image/object counts.

### `src.utils`

- `make_wavelengths(...)`: builds the wavelength axis after band trimming.
- `save_parquet(df, path, ...)`: writes summary and manifest tables with project-standard Parquet settings.

### `src.visualization`

- `plot_label_overlay_from_image_db(...)`: overlays segmentation labels on selected images.
- `plot_object_grid(...)`: displays multiple extracted objects for visual QC.
- `plot_object_view(...)`: displays a single extracted object and its associated mask/spectra.
- `build_database_inventory_table(...)`: creates the high-level database inventory by class, batch, and sample kind.

### `src.experiment_config`

- Stores shared workflow constants such as `DEFAULT_WAVELENGTH_MODE` and `DEFAULT_RESULTS_TAG`.

## Maintenance Checks

- The HDF5 file should be reloadable immediately after saving.
- HDF5 save should fail explicitly if an object-dtype array would otherwise be skipped or stored with loss.
- `image_summary.parquet` and `object_summary.parquet` should contain all expected pure, mixture, and reference images.
- Segmentation overlays should show one label per physical object, with no systematic missing objects or large merged regions.
- Any change to segmentation parameters should be documented and followed by rerunning notebooks 01, 02, and 03.
- If `OBJECT_MIN_AREA` changes, confirm that `image_summary.parquet` has `n_objects` values consistent with `n_labels` after the intended area filter.

## Automated Tests

- `tests/test_notebook00_database.py` covers `parse_image_key()`, split/object-label inference, object extraction metadata, `build_minimal_nir_uco_object_database()` area-threshold resolution from `segmentation_kwargs`, `detect_known_image_keys()`, `resolve_selected_keys()`, HDF5 validation, and the `save_nir_uco_h5()` / `load_nir_uco_h5()` roundtrip on a mini fixture.
- The HDF5 test verifies that compact object storage can reconstruct heavy arrays such as `mask_global`, `cube_crop`, and `image_ref_crop` on load.
