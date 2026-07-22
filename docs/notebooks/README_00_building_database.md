# Notebook 00 — Building the NIR UCO Database

Documentation for `00_building_database.ipynb`.

## Purpose

This notebook converts the raw NIR UCO hyperspectral data stored in `NIR_uco_sb.mat` into two structured databases:

- an **image-level database** (`image_db`) containing each preprocessed hyperspectral cube, its metadata, its segmentation mask, and its connected-component labels;
- an **object-level database** (`object_db`) containing one record per detected nut, including geometry, pixel spectra, summary spectra, and inherited sample metadata.

It then performs basic quality-control checks, saves compact tabular summaries, serializes both databases to HDF5, reloads the HDF5 file, and verifies that the reconstructed database is usable by downstream notebooks.

This is the first data-processing notebook in the project. Its primary output, `nir_uco_database.h5`, is the input database intended for later quality-control, exploratory analysis, preprocessing, and anomaly-detection notebooks.

> **Scope:** this notebook builds and validates the database. It does not train an anomaly-detection model or assign almond/peanut labels to individual objects in mixture images.

## Processing overview

The notebook executes the following pipeline:

1. Locate the project root and import the project modules.
2. Define raw-data paths, output paths, spectral settings, and segmentation parameters.
3. Load all variables from the MATLAB file.
4. Identify three-dimensional arrays that may be hyperspectral cubes.
5. Parse image names into standardized sample metadata.
6. Select either all recognized images or a user-defined subset.
7. Remove the first six noisy spectral bands and build the corresponding wavelength axis.
8. Segment foreground objects in each image.
9. Extract one object record per connected component.
10. Build and save image-, object-, and database-level summary tables.
11. Display segmentation and object-level quality-control plots.
12. Save the databases to a compressed HDF5 file.
13. Reload the HDF5 file and run a smoke test.

In compact form:

```text
NIR_uco_sb.mat
    -> load MATLAB variables
    -> recognize and parse image keys
    -> trim noisy bands
    -> create a 2D reference image
    -> threshold and clean the foreground mask
    -> label connected components
    -> extract object geometry and spectra
    -> image_db + object_db
    -> Parquet summaries + compressed HDF5 database
```

## Expected project layout

The notebook detects the project root when it is launched either from the project root or from a direct `notebooks/` child directory. The following files and directories are expected:

```text
project_root/
├── notebooks/
│   └── 00_building_database.ipynb
├── src/
│   ├── experiment_config.py
│   ├── utils.py
│   ├── data/
│   │   ├── database.py
│   │   └── segmentation.py
│   ├── io/
│   │   ├── dataload.py
│   │   └── database_h5.py
│   └── visualization/
│       ├── plot_images.py
│       ├── plot_objects.py
│       └── tables.py
├── HSI Data/
│   └── NIR camera UCO (889-1702 nm)/
│       └── NIR_uco_sb.mat
└── results/
```

The notebook creates `HSI Data/processed/` and `results/00_database/` if they do not already exist.

## Requirements

The recorded notebook kernel is named `hsi-nuts`. The code requires Python and the following packages:

- `numpy`
- `pandas`
- `scipy`
- `scikit-image`
- `h5py`
- `plotly`
- a Parquet engine supported by pandas, normally `pyarrow`
- JupyterLab or Jupyter Notebook

A minimal installation command is:

```bash
python -m pip install numpy pandas scipy scikit-image h5py plotly pyarrow jupyterlab
```

Use the environment and pinned versions supplied by the project, when available, instead of installing unpinned packages. The saved notebook output contains a `scikit-image` deprecation warning for `remove_small_objects(..., min_size=...)`; see [Known limitations and maintenance notes](#known-limitations-and-maintenance-notes).

## How to run the notebook

1. Place the raw MATLAB file at:

   ```text
   HSI Data/NIR camera UCO (889-1702 nm)/NIR_uco_sb.mat
   ```

   If the data are stored elsewhere, edit `RAW_MAT_PATH` in the parameters cell.

2. Activate the project environment.

3. Start Jupyter from the project root or from `notebooks/`:

   ```bash
   jupyter lab notebooks/00_building_database.ipynb
   ```

4. Review the parameters cell, especially the raw-data path, selected image keys, segmentation settings, and overwrite behavior.

5. Run all cells in order from top to bottom. Do not run only the save cells: the in-memory `image_db`, `object_db`, wavelength axis, and summary tables are created by earlier cells.

6. Inspect the segmentation overlays and object views before accepting the generated HDF5 file.

7. Confirm that the final cell reports successful completion and that the HDF5 reload smoke test passes.

The notebook uses `%autoreload 2`, so edits to imported project modules are automatically reloaded while the kernel is running. For a reproducible final build, restart the kernel and run all cells after modifying source code.

## Configuration reference

### Input and output settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `RAW_MAT_PATH` | `.../NIR_uco_sb.mat` | Raw MATLAB file containing image cubes and possibly non-image variables. |
| `PROCESSED_DIR` | `HSI Data/processed` | Directory for the reusable HDF5 database. |
| `RESULTS_DIR` | `results/00_database` | Directory for summary and manifest tables. |
| `SELECTED_KEYS` | `None` | Process every recognized three-dimensional image. A non-empty list restricts processing to those raw or clean keys. |
| `SKIP_UNKNOWN` | `True` | Ignore cubes whose names do not match the supported naming convention. |
| `OVERWRITE_OUTPUTS` | `True` | Permit replacement of existing primary outputs. |
| `INCLUDE_HEAVY_OBJECT_ARRAYS` | `False` | Omit redundant object arrays from HDF5 and reconstruct them when loading. |

`SELECTED_KEYS` accepts raw MATLAB keys such as `almond1_sb` and clean keys such as `almond1`. For example:

```python
SELECTED_KEYS = [
    "almond1",
    "peanut1",
    "alm1pea1",
]
```

An empty list is treated like `None` by the notebook's `if SELECTED_KEYS:` condition and therefore processes all automatically recognized images. Use a non-empty list to create a subset.

### Spectral settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `N_START` | `889` nm | Nominal wavelength of the first raw band. |
| `N_END` | `1702` nm | Nominal wavelength of the last raw band. |
| `N_BANDS_RAW` | `69` | Expected number of raw spectral bands. |
| `N_REMOVE_START` | `6` | Number of noisy bands removed from the beginning of every cube. |
| `N_STOP_END` | `None` | Optional exclusive upper band index; `None` retains all remaining bands. |
| `DATA_MODE` | `"reflectance"` | Semantic label stored with every image and object. No reflectance-to-absorbance conversion is performed here. |
| `WAVELENGTH_MODE` | `expcfg.DEFAULT_WAVELENGTH_MODE` | Project-wide label describing the wavelength selection. |
| `RESULTS_TAG` | `expcfg.DEFAULT_RESULTS_TAG` | Project-wide label stored in the manifest. |

`make_wavelengths` uses `numpy.linspace(889, 1702, 69)` and applies the same slice as the cubes. With the current settings, each cube changes from `(height, width, 69)` to `(height, width, 63)`, and the retained wavelength axis runs from approximately `960.74` to `1702.00` nm.

This wavelength construction assumes that the 69 bands are uniformly spaced between the two endpoint wavelengths. If calibrated wavelength values are available from the instrument, they should replace the generated linear axis.

### Split and label settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `FORCED_SPLIT` | `"projection"` | Explicit split assigned to every extracted object. |
| `OBJECT_MIN_AREA` | `10` pixels | Minimum connected-component area used during mask cleanup and object extraction. |

The explicit `FORCED_SPLIT="projection"` overrides the default split inference in `database.py`. Therefore, **all 1,262 objects in the recorded run, including pure almond and pure peanut objects, are assigned to `projection`**. Set `FORCED_SPLIT = None` to use the module's inferred roles:

- pure images: `train_minimal`;
- mixture images: `projection`;
- position-reference images: `position_reference`.

This choice affects downstream train/validation/projection logic and should be reviewed before model development.

### Segmentation settings

| Parameter | Current value | Effect |
| --- | ---: | --- |
| `reference_method` | `"max"` | For each pixel, use the maximum value across retained spectral bands to create the 2D reference image. |
| `threshold_method` | `"fixed"` | Mark pixels as foreground when the reference intensity is greater than the fixed threshold. |
| `tau_min` | `0.02` | Used as the fixed threshold because `tau` is not supplied. |
| `opening_radius` | `0` | Disable morphological opening. |
| `closing_radius` | `1` | Apply closing with a disk of radius one pixel. |
| `fill_holes` | `True` | Fill holes inside foreground regions. |
| `min_area` | `10` | Remove connected foreground components smaller than ten pixels; also used as the extraction minimum. |
| `use_watershed` | `False` | Use ordinary connected-component labeling rather than splitting touching objects. |
| `min_distance` | `10` | Minimum peak distance used only when watershed is enabled. |

The segmentation pipeline implemented by `segment_objects` is:

```python
image_ref = np.nanmax(cube, axis=2)
mask = image_ref > 0.02
mask = morphological_closing(mask, disk_radius=1)
mask = fill_holes(mask)
mask = remove_small_objects(mask, minimum_area=10)
labels = connected_component_labeling(mask)
```

If nuts touch and are merged into a single component, set `use_watershed=True` and tune `min_distance`. Any such change must be checked visually because watershed can also split a single nut into multiple objects.

### Quality-control settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `RUN_QC_PLOTS` | `True` | Generate interactive Plotly quality-control figures. |
| `N_QC_IMAGES` | `3` | Plot label overlays for the first three image summary rows. |
| `N_QC_OBJECTS` | `20` | Display at most twenty objects in the object grid. |

Set `RUN_QC_PLOTS=False` for a faster non-interactive build, but perform equivalent segmentation validation elsewhere before using the database.

## Image-name convention and metadata parsing

`parse_image_key` lowercases names, removes known suffixes such as `_sb`, and recognizes the following patterns:

| Sample type | Examples | Parsed metadata |
| --- | --- | --- |
| Pure sample | `almond1_sb`, `alm1`, `peanut2_sb`, `pea2` | One `nut_type`, one `batch`, `is_pure=True`. |
| Mixture | `alm1pea2_sb`, `alm1pea2wal3` | A `components` dictionary with one batch per nut, `nut_type="mixture"`, `is_mixture=True`. |
| Position reference | `pea1_pos3_sb`, `peanut2_pos4` | One nut type, one batch, one `position_set`, `is_position_reference=True`. |

Configured aliases are:

- `almond` and `alm` -> `almond`;
- `peanut` and `pea` -> `peanut`;
- `walnut` and `wal` -> `walnut`.

Unknown nut tokens, malformed names, duplicate mixture components, and names containing unsupported extra text are returned with `sample_kind="unknown"`. They are excluded when `SKIP_UNKNOWN=True`.

Object labels are inferred conservatively:

- objects from pure images inherit the known nut type;
- objects from position-reference images inherit the known nut type;
- objects from mixture images receive `object_nut_type="unknown"`, because segmentation determines object boundaries but not class identity.

The latter is fundamental to the anomaly-detection use case: mixture objects must be classified or scored by a later model rather than labeled from the mixture filename.

## Detailed notebook walkthrough

Cell numbers below are zero-based, matching the cell positions in the `.ipynb` file.

### Cell 0 — Notebook objective

Introduces the image-level and object-level databases and lists the intended persisted outputs. The current implementation saves summaries as **Parquet**, not CSV.

### Cell 1 — Imports and project-root detection

Imports standard Python, NumPy, and pandas utilities; expands pandas display limits; and determines `PROJECT_ROOT` by looking for a `src/` directory in the current directory or its parent. It inserts the project root at the beginning of `sys.path`, allowing imports such as `from src.io.dataload import load_mat_file`.

A `RuntimeError` is raised if `src/` cannot be found at either level. Launching the notebook from a deeper or unrelated directory requires adjusting the root-detection logic.

### Cells 2–3 — Project imports and automatic reloading

Imports the data-loading, metadata-parsing, preprocessing, database-building, HDF5, table, and visualization functions. `%autoreload 2` reloads imported modules before code execution, which is convenient during development.

### Cell 4 — Paths and processing parameters

Defines every path and high-level parameter. Output directories are created immediately with `mkdir(parents=True, exist_ok=True)`. This is the main cell users should edit before executing the pipeline.

### Cell 5 — Input and overwrite safeguards

Stops execution if the raw MATLAB file does not exist. It also checks whether the primary HDF5, image-summary, object-summary, or manifest outputs already exist. If any exist while `OVERWRITE_OUTPUTS=False`, the notebook raises `FileExistsError` before loading the dataset.

`database_inventory.parquet` is not included in this preflight list, although it is also written later.

### Cell 6 — MATLAB loading

`load_mat_file` first calls `scipy.io.loadmat`, which handles traditional MATLAB files, and removes internal keys beginning with `__`. If SciPy raises `NotImplementedError`, the function treats the file as HDF5 (the storage format commonly used by MATLAB v7.3) and recursively loads every HDF5 dataset with `h5py`.

The cell records load time, entry count, and the first variable names. In the saved reference run, the file contained 48 entries and loaded in 12.82 seconds.

### Cell 7 — Raw-entry inventory

Converts every loaded value with `numpy.asarray` and records its Python type, number of dimensions, shape, dtype, and whether it is a candidate cube. At this stage, a candidate cube means only `ndim == 3`; spectral size and name validity are checked separately.

This table is useful for detecting unexpected metadata arrays, transposed cubes, or extra variables in the MATLAB file.

### Cell 8 — Candidate-name parsing

Calls `parse_image_key` for every three-dimensional entry and constructs a human-readable table containing sample kind, nut type, batch, position set, flags, description, and cube dimensions. Unknown patterns remain visible in this diagnostic table.

### Cell 9 — Image selection

If `SELECTED_KEYS` is a non-empty list, `resolve_selected_keys` accepts both raw and suffix-free names and raises `KeyError` for unresolved names. Otherwise, `detect_known_image_keys` selects three-dimensional NumPy arrays whose names are recognized by `parse_image_key`.

The cell raises `RuntimeError` if selection returns no images. The selected metadata table should be reviewed before database construction.

### Cell 10 — Dataset-composition summaries

Groups the parsed candidate table by sample kind, nut type, and batch. This describes all candidate cubes, not only a manually selected subset. With `SELECTED_KEYS` set, use `detected_df` if a summary strictly limited to selected images is required.

### Cell 11 — Wavelength-axis construction

Builds a uniformly spaced 69-band axis from 889 to 1702 nm and removes the same leading bands that will be removed from the cubes. The current result contains 63 wavelengths.

### Cell 12 — Band and value-range validation

Preprocesses each selected cube with `preprocess_nir_uco_cube`, records raw and cleaned shapes and extrema, and verifies that the wavelength count matches the cleaned cube's band count.

`preprocess_nir_uco_cube` currently performs spectral slicing only. It does not normalize, smooth, standardize, convert reflectance to absorbance, remove spatial background, or impute non-finite values.

The wavelength-length check uses the first cleaned cube as the expected band count. The displayed table should therefore also be inspected for inconsistent shapes among later cubes.

### Cell 13 — Database construction

Calls `build_minimal_nir_uco_object_database`. For each selected image, that function:

1. verifies that the key exists and the value is a three-dimensional NumPy array;
2. parses the image name;
3. skips unknown names when requested;
4. removes noisy spectral bands;
5. calls `segment_objects`;
6. validates that the cube, reference image, mask, and label image have compatible dimensions;
7. calculates segmentation metadata such as foreground area and positive-label count;
8. calls `extract_objects_from_labeled_image`;
9. appends the extracted records to `object_db`;
10. stores the image record in `image_db` under its clean key.

Object extraction uses `skimage.measure.regionprops`. For every region at or above the extraction minimum area, it computes a bounding box, centroid, cropped and global masks, global and local pixel positions, reference-image crop, cube crop, and a spectral matrix:

```text
spectra.shape = (number of object pixels, number of retained bands)
```

It then computes the per-band mean, median, and standard deviation with NaN-aware NumPy functions. Object identifiers follow the pattern `<clean_image_key>_objNNN`, for example `alm1pea1_obj001`.

### Cell 14 — Database inventory

`build_database_inventory_table` summarizes the number of images, source images, detected objects, object pixels, and mean/median object area by nut type, batch, and sample kind. The result is saved as `database_inventory.parquet`.

Image-level mixtures have `nut_type="mixture"`, whereas their object records have `object_nut_type="unknown"`. Consequently, the outer merge can produce separate mixture-image and unknown-object rows. This is expected from the current schema, not necessarily missing data.

### Cell 15 — Compatibility and area checks

Prints the key legacy-compatible parameters, displays the split distribution, calculates minimum/median/maximum object areas, and warns if any extracted object is smaller than `OBJECT_MIN_AREA`.

The same minimum area is applied during mask cleanup and extraction. Rechecking at extraction protects against inconsistent upstream label images and future changes to the segmentation implementation.

### Cells 16–17 — Flat summary tables

Transform the nested dictionaries into two flat pandas DataFrames:

- `image_summary_df`: one row per image;
- `object_summary_df`: one row per detected object.

Large arrays are deliberately excluded, making these tables suitable for fast inspection and later joins.

### Cell 18 — Aggregate quality-control statistics

Reports image and object counts by sample kind and nut type, followed by object counts and median areas by batch. Large differences in counts or areas can reveal missed objects, merged nuts, fragmented masks, inconsistent acquisition conditions, or inappropriate thresholds.

### Cell 19 — Save essential summaries

Uses `save_parquet`, which converts nested values to JSON strings when necessary, optionally downcasts numeric types, converts suitable low-cardinality columns to categorical dtype, and writes Zstandard-compressed Parquet files through pandas.

### Cell 20 — Segmentation overlays

For the first `N_QC_IMAGES` images, overlays nonzero label IDs on `image_ref` and crops the Plotly view to the object foreground plus five pixels of padding. Each visually distinct nut should generally correspond to one label color.

### Cell 21 — Object-level plots

Selects the first image containing objects, displays up to `N_QC_OBJECTS` object crops sorted by area, and then displays one object with its mask, mean spectrum, and a plus-or-minus-one-standard-deviation envelope.

### Cell 22 — HDF5 serialization

When overwriting is enabled, explicitly deletes the existing HDF5 file before calling `save_nir_uco_h5`. The HDF5 writer creates top-level `images` and `objects` groups, records format and count attributes, saves arrays as datasets, and saves scalar or JSON-compatible metadata as group attributes.

Arrays larger than 1,000 elements use Gzip compression level 4 with the HDF5 shuffle filter. With `INCLUDE_HEAVY_OBJECT_ARRAYS=False`, the redundant per-object `mask_global`, `cube_crop`, and `image_ref_crop` arrays are omitted.

### Cell 23 — Reload smoke test

`load_nir_uco_h5` validates the file format, required groups, and stored record counts. It reloads image and object records and, by default, reconstructs omitted heavy object arrays from each object's bounding box and the source image data.

The smoke test compares original and reloaded image/object counts, then verifies that one reloaded object contains identification, label, mask, spectral, position, geometry, and summary-spectrum fields. This checks basic serialization integrity, but it is not a full value-by-value comparison of every record.

### Cell 24 — Reproducibility manifest

Writes a one-row manifest containing source and output paths, wavelength settings, band-selection settings, segmentation minimum area, split policy, HDF5 storage mode, and final image/object counts.

### Cell 25 — Completion message

Lists the essential outputs and identifies `01_database_quality_check.ipynb` as the next notebook.

## In-memory database schemas

### `image_db`

`image_db` is a dictionary indexed by the clean image key, for example `image_db["almond1"]`. Important fields include:

| Field | Type or shape | Description |
| --- | --- | --- |
| `image_id` | string | Original MATLAB variable name, including a suffix such as `_sb`. |
| `clean_key` | string | Normalized key used as the dictionary key. |
| `sample_kind` | string | `pure`, `mixture`, or `position_reference`. |
| `nut_type` | string | Known image class or `mixture`. |
| `batch` | integer or `None` | Batch for pure and position-reference images. Mixture component batches are stored in `components`. |
| `components` | dictionary | Nut types, aliases, and batches parsed from the filename. |
| `position_set` | integer or `None` | Position-reference acquisition set. |
| `cube` | `(H, W, B)` array | Preprocessed hyperspectral cube. |
| `image_ref` | `(H, W)` array | 2D image used for segmentation. |
| `mask` | `(H, W)` Boolean array | Clean foreground mask. |
| `labels` | `(H, W)` integer array | `0` for background and positive integers for detected components. |
| `threshold` | float | Segmentation threshold actually used. |
| `segmentation` | dictionary | Label count, foreground area, area ratio, and segmentation shape. |
| `wavelengths` | `(B,)` array | Retained wavelength values in nanometres. |
| `data_mode` | string | Semantic intensity mode, currently `reflectance`. |
| `n_objects` | integer | Number of objects retained after extraction filtering. |
| `object_ids` | list of strings | References to the associated `object_db` records. |

The parsed Boolean flags and human-readable description are also retained.

### `object_db`

`object_db` is a dictionary indexed by object ID, for example `object_db["almond1_obj001"]`. Its fields fall into five groups:

| Group | Fields | Description |
| --- | --- | --- |
| Identification | `object_id`, `object_index`, `label_id` | Stable object key, per-image extraction order, and connected-component label. |
| Provenance and labels | `source_image`, `source_clean_key`, `sample_kind`, `image_nut_type`, `object_nut_type`, `batch`, `components`, `position_set`, flags, `split` | Image origin, parsed metadata, known/unknown object class, and downstream role. |
| Geometry | `bbox`, `centroid`, `area_pixels`, `n_pixels` | Region extent and size. `bbox` is `(min_row, min_col, max_row, max_col)` with exclusive maxima. |
| Spatial arrays | `mask`, `mask_global`, `positions_global`, `positions_local`, `cube_crop`, `image_ref_crop` | Cropped/global representations and pixel coordinates. |
| Spectral arrays | `spectra`, `mean_spectrum`, `median_spectrum`, `std_spectrum`, `wavelengths`, `n_bands`, `data_mode` | Per-pixel spectra and per-band summary statistics. |

`area_pixels` and `n_pixels` should describe the same detected region. `spectra.shape[0]` equals `n_pixels`; `spectra.shape[1]` equals `n_bands`.

## Persisted outputs

| Output | Purpose |
| --- | --- |
| `HSI Data/processed/nir_uco_database.h5` | Reusable image- and object-level database for downstream notebooks. |
| `results/00_database/image_summary.parquet` | One lightweight row per image. |
| `results/00_database/object_summary.parquet` | One lightweight row per detected object. |
| `results/00_database/database_manifest.parquet` | One-row record of build paths, settings, and final counts. |
| `results/00_database/database_inventory.parquet` | Aggregated inventory by class, batch, and sample kind. |

Despite the introductory markdown cell's reference to “CSV summaries,” the current code writes Parquet files only.

### HDF5 layout

The HDF5 file uses format identifier `nir_uco_object_database` and version `1.0`:

```text
nir_uco_database.h5
├── attributes: format, version, n_images, n_objects,
│               include_heavy_object_arrays
├── images/
│   └── <clean_image_key>/
│       ├── datasets: cube, image_ref, mask, labels, wavelengths
│       └── attributes: remaining image metadata
└── objects/
    └── <object_id>/
        ├── datasets: mask, positions, spectra, summary spectra, wavelengths
        └── attributes: remaining object metadata
```

Dictionary, list, tuple, and set metadata are JSON-encoded in HDF5 attributes. `None` uses an internal sentinel. Tuple-like `bbox` and `centroid` fields are restored as tuples when loading.

To load the database in another notebook:

```python
from pathlib import Path
from src.io.database_h5 import load_nir_uco_h5

db_path = Path("HSI Data/processed/nir_uco_database.h5")
object_db, image_db = load_nir_uco_h5(
    db_path,
    reconstruct_heavy_object_arrays=True,
)
```

Set `reconstruct_heavy_object_arrays=False` when only compact spectral and metadata fields are needed and memory use matters.

## Reference run recorded in the notebook

The saved notebook outputs document the following run. These values are reference results, not hard-coded acceptance criteria for future datasets or parameter changes.

| Metric | Recorded value |
| --- | ---: |
| Raw MATLAB entries / selected images | 48 / 48 |
| Raw cube shape | `(370, 318, 69)` |
| Clean cube shape | `(370, 318, 63)` |
| Mixture images | 20 |
| Position-reference images | 20 |
| Pure almond images | 4 |
| Pure peanut images | 4 |
| Total extracted objects | 1,262 |
| Mixture objects with unknown class | 722 |
| Position-reference peanut objects | 146 |
| Pure almond objects | 214 |
| Pure peanut objects | 180 |
| Object area, minimum / median / maximum | 12 / 77 / 224 pixels |
| HDF5 size with compact object storage | 115.63 MB |

The recorded position-reference counts are highly uneven: several images contain 15 detected objects, while multiple position sets contain only one. This may reflect the acquisition design, touching objects merged because watershed is disabled, or a segmentation issue. The overlays must be inspected before these records are used as independent object samples.

## Quality-control checklist

Before accepting a generated database, verify all of the following:

- Every intended MATLAB key appears in the selected-image table.
- No unexpected three-dimensional variables have been interpreted as images.
- Parsed sample kind, nut type, batch, mixture components, and position set match the acquisition log.
- Every cleaned cube has the expected number of bands.
- The wavelength count equals the spectral dimension.
- Cleaned value ranges are plausible for the declared `DATA_MODE`.
- Each visible nut is covered by a mask.
- Background regions are not retained as objects.
- Touching nuts are not unintentionally merged.
- Single nuts are not fragmented into several labels.
- Object counts and area distributions are plausible across batches and acquisition types.
- The split distribution matches the intended downstream experiment.
- The HDF5 reload count assertions and required-field smoke test pass.

## Common modifications

### Process only a small test subset

Use a representative pure almond image, pure peanut image, and mixture before running the full dataset:

```python
SELECTED_KEYS = ["almond1", "peanut1", "alm1pea1"]
RUN_QC_PLOTS = True
```

Remember that this overwrites the standard output paths when `OVERWRITE_OUTPUTS=True`. Use temporary output filenames if the full database must be preserved.

### Change the threshold strategy

Supported values in `segment_objects` are:

- `fixed`: use `tau` when supplied, otherwise `tau_min`;
- `otsu`: estimate the threshold from finite reference-image pixels;
- `otsu_min`: use the larger of the Otsu threshold and `tau_min`;
- `percentile`: threshold at the configured image-intensity percentile.

For example:

```python
SEGMENTATION_KWARGS.update({
    "threshold_method": "otsu_min",
    "tau_min": 0.02,
})
```

Threshold changes should be evaluated across all sample kinds and batches rather than on a single visually convenient image.

### Split touching objects

```python
SEGMENTATION_KWARGS.update({
    "use_watershed": True,
    "min_distance": 10,
})
```

Reduce `min_distance` to allow closer watershed markers; increase it to discourage over-segmentation.

### Store all object arrays explicitly

```python
INCLUDE_HEAVY_OBJECT_ARRAYS = True
```

This stores `mask_global`, `cube_crop`, and `image_ref_crop` for every object. It makes the file larger and duplicates information already present at image level, but avoids reconstruction during loading.

## Troubleshooting

### `RuntimeError: Could not find project root`

Start Jupyter from the project root or its direct `notebooks/` directory. Confirm that `src/` exists at the detected root.

### `FileNotFoundError: Raw .mat file not found`

Check the filename, capitalization, folder structure, and `RAW_MAT_PATH`. Avoid embedding a machine-specific absolute path in the committed notebook.

### No recognized image is found

Inspect `raw_entries_df` and `parsed_df`. Confirm that cubes are three-dimensional and that names match the supported patterns. Add aliases or patterns to `NIR_UCO_NAME_CONFIG` only when the naming convention is well defined.

### A selected key is not found

Use either the exact raw key or its clean suffix-free equivalent. `resolve_selected_keys` reports every unresolved key in one `KeyError`.

### Wavelength length does not match the cube

Verify `N_BANDS_RAW`, `N_REMOVE_START`, and `N_STOP_END`, and confirm that every raw cube uses the same spectral axis. Update cube slicing and wavelength slicing together.

### Too few objects are detected

Inspect the reference image and label overlay. Possible causes include an excessively high threshold, an excessive minimum area, nuts connected by mask bridges, or unexpected data scaling. Consider `otsu_min`, a smaller `tau_min`, less closing, or watershed, one change at a time.

### Too many objects are detected

Possible causes include a low threshold, noisy foreground pixels, insufficient morphological cleaning, or watershed over-segmentation. Consider increasing the threshold or minimum area, applying a small opening radius, or increasing watershed `min_distance`.

### HDF5 cannot be written

Confirm that the destination directory is writable and that no other process holds the file open. Object-dtype NumPy arrays cannot be written as HDF5 datasets by this serializer without an explicit conversion.

### Parquet export fails

Install `pyarrow` or another pandas-compatible Parquet engine. The helper converts common nested values to JSON strings, but unsupported custom objects may still require explicit serialization.

### Memory usage is high after loading

The image database contains every cleaned hyperspectral cube. In addition, `reconstruct_heavy_object_arrays=True` recreates global masks and crops for every object. Disable reconstruction when those fields are unnecessary, load only summary Parquet files for tabular analysis, or design a lazy-loading layer for larger datasets.

## Known limitations and maintenance notes

- The initial notebook description says that CSV summaries are produced, but the implementation produces Parquet files.
- `database_inventory.parquet` is written but is absent from the initial overwrite preflight and the final “Essential outputs” list.
- The wavelength axis is inferred from two endpoints and an assumed uniform spacing rather than read from instrument calibration metadata.
- `DATA_MODE="reflectance"` is metadata only; the notebook does not verify physical calibration or convert units.
- Automatic image selection checks array dimensionality and filename validity, but not a required spectral band count.
- The batch summaries in Cell 10 are based on all parsed candidate cubes, not necessarily the manually selected subset.
- `FORCED_SPLIT="projection"` overrides metadata-based split inference for all objects.
- Connected-component labeling does not separate touching nuts when `use_watershed=False`.
- The HDF5 smoke test checks counts and required fields but does not compare all arrays and metadata values against the original databases.
- The saved execution emits a `scikit-image` `FutureWarning` because the `min_size` parameter of `morphology.remove_small_objects` is deprecated in the installed version. Pin a compatible `scikit-image` release or update `clean_mask` after confirming the replacement API's boundary semantics; the warning states that the newer threshold behavior differs for objects exactly equal to the limit.
- The notebook deletes an existing HDF5 file with `Path.unlink()` when overwriting is enabled. Keep backups of validated databases or use versioned output names when experimenting.

## Downstream use

Once this notebook has completed and the database has passed visual and statistical quality control, continue with:

```text
01_database_quality_check.ipynb
```

Downstream code should normally load `nir_uco_database.h5` through `load_nir_uco_h5` rather than re-running raw-data ingestion. Use the Parquet summaries when only metadata, counts, geometry, or manifest settings are needed.
