# Notebook 01 — NIR UCO Database Quality Check

Documentation for `01_database_quality_check.ipynb`.

## Purpose

This notebook performs structural, statistical, visual, and spectral quality control on the NIR UCO database created by `00_building_database.ipynb`.

It loads the persisted HDF5 database without rebuilding it and examines:

- image-level metadata and segmentation summaries;
- object-level metadata, geometry, and spectral dimensions;
- segmentation overlays and object crops;
- object-count and object-area distributions;
- pure almond and peanut spectra across batches;
- simple spectral outlier candidates;
- the presence of fields required by downstream notebooks;
- consistency between object arrays, recorded dimensions, and source-image cubes.

The primary persisted results are lightweight Parquet quality-control tables. Interactive Plotly figures are displayed in the notebook for human review but are not exported by the current implementation.

> **Scope:** this notebook diagnoses an existing database. It does not modify the HDF5 database, rerun segmentation, remove suspicious records, relabel mixture objects, or train an anomaly-detection model.

> **Important:** an empty QC flag table means that none of the implemented structural warning rules fired. It does not, by itself, prove that segmentation, sample labeling, acquisition quality, or spectral behavior is scientifically valid.

## Relationship to the previous notebook

The expected sequence is:

```text
00_building_database.ipynb
    -> HSI Data/processed/nir_uco_database.h5
    -> 01_database_quality_check.ipynb
    -> results/01_quality_check/*.parquet
    -> 02_matrices_preprocessing.ipynb
```

Notebook 01 intentionally loads the serialized database rather than relying on `image_db` or `object_db` left in memory by Notebook 00. This tests that the persisted artifact is independently reusable.

## Quality-control overview

The notebook follows these stages:

1. Locate the project root and import project functions.
2. Configure the input HDF5 path, optional raw-data comparison, output paths, and plotting switches.
3. Load `image_db` and `object_db` from HDF5.
4. Build and save a flat image-level QC table.
5. Summarize image counts, object counts, and segmentation-mask coverage.
6. Build image-level structural warnings.
7. Build and save a flat object-level QC table.
8. Summarize object counts and area distributions by sample kind, class, and batch.
9. Build object-level structural warnings.
10. Select deterministic representative images and inspect segmentation overlays.
11. Inspect object grids and detailed object spectra.
12. Build spectral summary statistics and plot class/batch behavior.
13. Rank objects with a simple global spectral QC score.
14. Inspect pixel-level spectral distributions in representative images.
15. Check required fields and array-shape consistency.
16. Combine structural findings into a canonical QC flag table.
17. Summarize data availability for the next modelling stage.

## Expected project layout

The notebook must be launched from the project root or its direct `notebooks/` child directory:

```text
project_root/
├── notebooks/
│   ├── 00_building_database.ipynb
│   └── 01_database_quality_check.ipynb
├── src/
│   ├── experiment_config.py
│   ├── utils.py
│   ├── data/
│   │   └── database.py
│   ├── io/
│   │   ├── dataload.py
│   │   └── database_h5.py
│   ├── workflows/
│   │   └── quality_check.py
│   └── visualization/
│       ├── plot_generic.py
│       ├── plot_images.py
│       ├── plot_objects.py
│       └── plot_spectra.py
├── HSI Data/
│   ├── processed/
│   │   └── nir_uco_database.h5
│   └── NIR camera UCO (889-1702 nm)/
│       └── NIR_uco_sb.mat            # required only for optional comparison
└── results/
```

The notebook creates `results/01_quality_check/` automatically. It never creates or replaces `nir_uco_database.h5`.

## Requirements

The recorded notebook kernel is named `hsi-nuts`. Use the same project environment as Notebook 00. The directly or indirectly required packages include:

- `numpy`
- `pandas`
- `scipy`
- `scikit-image`
- `h5py`
- `plotly`
- a pandas-compatible Parquet engine, normally `pyarrow`
- JupyterLab or Jupyter Notebook

A minimal installation command is:

```bash
python -m pip install numpy pandas scipy scikit-image h5py plotly pyarrow jupyterlab
```

Prefer the project's pinned environment when one is provided. Using the same dependency versions for database construction and quality control reduces serialization and plotting differences.

## How to run the notebook

1. Run `00_building_database.ipynb` and verify that the following file exists:

   ```text
   HSI Data/processed/nir_uco_database.h5
   ```

2. Activate the project environment.

3. Start Jupyter from the project root or from `notebooks/`:

   ```bash
   jupyter lab notebooks/01_database_quality_check.ipynb
   ```

4. Review the parameter cell. Decide whether heavy object arrays must be reconstructed and which interactive plot families should run.

5. Restart the kernel and run all cells in order.

6. Do not judge the result from the final warning counts alone. Inspect the image-count tables, area distributions, segmentation overlays, object grids, spectra, and ranked spectral candidates.

7. Continue to `02_matrices_preprocessing.ipynb` only after the structural checks pass and the visual/scientific findings are acceptable.

The notebook uses `%autoreload 2`, so imported project modules are refreshed during development. For a reproducible QC run, restart the kernel and execute all cells after changing any source module.

## Configuration reference

### Input settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `DB_H5_PATH` | `HSI Data/processed/nir_uco_database.h5` | HDF5 database produced by Notebook 00. |
| `RAW_MAT_PATH` | `.../NIR_uco_sb.mat` | Raw MATLAB file used only when `RUN_RAW_DB_COMPARISON=True`. |
| `N_REMOVE_START` | `6` | Number of leading bands removed during the optional raw comparison. |
| `N_STOP_END` | `None` | Optional exclusive final band index used during the raw comparison. |
| `RECONSTRUCT_HEAVY_OBJECT_ARRAYS` | `True` | Recreate omitted `mask_global`, `cube_crop`, and `image_ref_crop` object arrays while loading. |

Reconstruction uses each object's bounding box and cropped mask together with the source image's `cube`, `image_ref`, and `labels`. It restores compatibility with object visualizations and the default required-field checks, at the cost of additional memory.

If `RECONSTRUCT_HEAVY_OBJECT_ARRAYS=False`:

- `mask_global`, `cube_crop`, and `image_ref_crop` may remain absent because Notebook 00 used compact HDF5 storage;
- object-grid and detailed-object plots may fail because they expect `image_ref_crop`;
- `check_missing_required_fields` will flag `mask_global` as missing under the default schema;
- compact metadata and spectral analyses can still work if their required fields are present.

### Output settings

| Parameter | Current path | Meaning |
| --- | --- | --- |
| `RESULTS_DIR` | `results/01_quality_check` | Destination directory for persisted QC tables. |
| `IMAGE_QC_PATH` | `image_qc_summary.parquet` | One row per image. Always written after successful table construction. |
| `OBJECT_QC_PATH` | `object_qc_summary.parquet` | One row per object. Always written after successful table construction. |
| `QC_FLAGS_PATH` | `qc_flags.parquet` | Canonical structural warning table, written only when at least one flag exists. |

There is no overwrite guard. `save_parquet` replaces existing image and object QC summaries at the same paths.

### Plot switches

| Parameter | Current value | What it controls |
| --- | ---: | --- |
| `RUN_SEGMENTATION_QC_PLOTS` | `True` | Segmentation label overlays in Cell 23. |
| `RUN_OBJECT_QC_PLOTS` | `True` | Per-image object grids in Cell 27. |
| `RUN_SPECTRAL_QC_PLOTS` | `True` | Detailed views of selected individual objects in Cell 29 only. |
| `RUN_RAW_DB_COMPARISON` | `False` | Optional raw-versus-stored reference-image comparison in Cell 25. |

`RUN_SPECTRAL_QC_PLOTS` does **not** disable the later spectral plots in Cells 34–41. Those cells run unconditionally in the current notebook. To perform a genuinely non-graphical run, skip or guard those cells explicitly.

### Sampling and display settings

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `N_EXAMPLE_IMAGES_PER_KIND` | `2` | Number of representative images sampled from each `sample_kind`. |
| `N_OBJECTS_IN_GRID` | `25` | Maximum objects shown in each object grid. |
| `RANDOM_STATE` | `expcfg.RANDOM_STATE` | Seed used for representative-image, pure-spectrum, and pixel sampling. The recorded configuration uses `42`. |
| `MAX_OBJECT_SPECTRA_PER_GROUP` | `80` | Maximum sampled pure-object mean spectra per class in Cell 34. |

Representative images are sampled by `sample_kind`, not by every nut type, batch, or position set. Two images per kind therefore do not guarantee full coverage of all experimental groups.

### Raw comparison settings

The parameter cell defines `SEGMENTATION_KWARGS`, mirroring Notebook 00, but the dictionary is never used by this notebook. Enabling `RUN_RAW_DB_COMPARISON` does not rerun `segment_objects` and does not validate those segmentation parameters.

The raw comparison uses hard-coded keys:

```python
RAW_KEY = "pea4_pos4_sb"
DB_KEY = "pea4_pos4"
```

It removes the configured noisy bands, recomputes the maximum-across-bands reference image, and displays the raw-derived and stored reference images separately. Change both keys together to inspect another image.

## Implemented QC rules

### Image-level warnings

`build_image_qc_warnings` creates one row per triggered condition:

| Condition | Warning text | Interpretation |
| --- | --- | --- |
| `n_objects_recorded == 0` | `No object detected` | The image record contains no extracted objects. |
| `n_objects_recorded != n_labels_positive` | Counts differ | The stored object count does not match the number of positive labels. |
| `mask_area_ratio` is missing or `<= 0` | `Empty or invalid mask area ratio` | The image has no valid positive foreground coverage. |

The function does not currently warn about unusually low or high nonzero mask coverage, extreme object counts, missing wavelength axes, inconsistent thresholds, unexpected data modes, or batch imbalance.

### Object-level warnings

`build_object_qc_warnings` checks:

| Condition | Meaning |
| --- | --- |
| `area_pixels` is missing, non-finite, or `<= 0` | Invalid object area. |
| `n_pixels != area_pixels` | The spectral-pixel count and region area disagree. |
| `mean_spectrum_length != n_bands` | The mean spectrum does not have the recorded spectral dimension. |
| `bbox_area < area_pixels` | The object contains more pixels than can fit in its bounding box. |

These rules are structural. They do not flag unusually small or large but positive objects, non-finite values inside the full pixel-level `spectra` matrix, implausible centroids, overlapping objects, incorrect class labels, or spectral outliers.

### Required image fields

The default image schema requires:

```text
cube, image_ref, mask, labels, clean_key, sample_kind,
nut_type, n_objects, object_ids
```

### Required object fields

The default object schema requires:

```text
object_id, source_clean_key, sample_kind, object_nut_type,
batch, split, bbox, centroid, area_pixels, mask, mask_global,
positions_global, spectra, mean_spectrum, median_spectrum,
std_spectrum
```

`check_missing_required_fields` tests key presence, not whether values are non-null, finite, correctly typed, or nonempty.

### Object shape checks

`build_object_shape_check_tables` checks six relationships for every object:

```text
spectra.shape[0]          == n_pixels
positions_global.shape[0] == n_pixels
len(mean_spectrum)         == n_bands
len(median_spectrum)       == n_bands
len(std_spectrum)          == n_bands
object n_bands             == source image cube.shape[2]
```

The function returns both the complete `shape_check_df` and the failing subset `bad_shape_df`. Only the failing subset is displayed and incorporated into `qc_flags_df`.

### Canonical QC flags

`build_qc_flags_table` combines four sources into the columns:

| Column | Meaning |
| --- | --- |
| `record_type` | `image` or `object`. |
| `record_id` | Clean image key or object ID. |
| `flag_type` | `image_warning`, `object_warning`, `missing_fields`, or `bad_shape`. |
| `warning` | Human-readable finding. |

Multiple flags can exist for the same record. The table is empty when no implemented rule fires.

## Detailed notebook walkthrough

Cell numbers are zero-based and correspond to positions in the `.ipynb` file.

### Cell 0 — Scope and objectives

States that the notebook loads the saved database and checks metadata, segmentation, area distributions, spectra, batch effects, and noisy images. The text refers to `00_build_database.ipynb`, but the supplied construction notebook is named `00_building_database.ipynb`.

### Cell 1 — Imports and project-root detection

Imports `Path`, NumPy, and pandas; expands DataFrame display limits; finds `PROJECT_ROOT` by looking for `src/` in the current directory or its parent; and inserts the project root into `sys.path`.

Running from an unrelated or deeper directory raises `RuntimeError`.

### Cells 2–3 — Project imports and autoreload

Imports HDF5 loading, optional MATLAB loading and cube preprocessing, experiment configuration, plotting utilities, Parquet helpers, and all QC workflow functions. `%autoreload 2` refreshes changed project modules automatically.

### Cell 4 — Configuration

Defines the HDF5 input, optional raw input, band slicing, an unused segmentation dictionary, output paths, plot switches, heavy-array reconstruction, sample sizes, and random seed. It creates `results/01_quality_check/` if necessary.

### Cell 5 — Database existence check

Raises `FileNotFoundError` if the HDF5 file is absent and otherwise reports its size. The error message again refers to `00_build_database.ipynb`; the actual supplied notebook is `00_building_database.ipynb`.

### Cell 6 — HDF5 loading

Calls `load_nir_uco_h5`, which validates the HDF5 format, required top-level groups, and stored record counts before returning `object_db` and `image_db`. With reconstruction enabled, redundant heavy object arrays omitted during saving are recreated.

The cell prints database sizes and sample keys so an unexpected file or empty database is immediately visible.

### Cell 7 — Image QC table

`build_image_qc_table` constructs one row per image and sorts by sample kind, nut type, batch, position set, and clean key. It derives spatial dimensions from `cube`, counts positive unique labels, calculates image pixel count and mask coverage, checks for a nonempty wavelength axis, and copies relevant metadata.

Core fields such as `cube` and `labels` are accessed directly. If they are missing, table construction can fail before the later required-field check runs.

### Cell 8 — Save image QC table

Writes `image_qc_df` as compressed Parquet through the shared `save_parquet` helper.

### Cells 9–10 — Image summaries

Cell 9 groups by sample kind and nut type to report image counts, total objects, and minimum/median/maximum objects per image. Cell 10 adds batch and median mask-area ratio.

These summaries are displayed but not saved as separate files.

### Cell 11 — Image-count plots

Creates:

- grouped counts by sample kind and nut type;
- one bar per image showing `n_objects_recorded`.

`plot_counts_by_group` computes counts from DataFrame rows, creates one trace per group, and uses stable project colors. `plot_bar_values` retains image-level detail but may become visually dense for larger databases.

### Cell 12 — Image warnings

Applies the three image warning rules described above. The result is kept in memory for the consolidated flag table.

### Cell 13 — Object-QC section introduction

Documents the intended metadata and geometry checks.

### Cell 14 — Object QC table

`build_object_qc_table` constructs one row per object. It records provenance, labels, flags, split, area, spectral dimensions, centroid, bounding box, bounding-box dimensions, and data mode.

`bbox_area` is the area of the enclosing rectangle, not the segmented object. It should be greater than or equal to `area_pixels`.

The helper expects usable `spectra`, `mean_spectrum`, `centroid`, and `bbox` values. Severely malformed records can cause this table-building stage to fail before dedicated missing-field diagnostics are reached.

### Cell 15 — Save object QC table

Writes `object_qc_df` as compressed Parquet.

### Cells 16–18 — Object and area summaries

Cell 16 summarizes counts and areas by sample kind and object label. Cell 17 adds batch. Cell 18 calculates group size, mean, standard deviation, minimum, 5th percentile, median, 95th percentile, and maximum area.

The area summaries are diagnostic displays only and are not separately persisted.

### Cell 19 — Object-count and area plots

Displays grouped object counts and a batch-faceted box plot of object areas. Only statistical outliers selected by the box-plot convention are drawn as individual points.

Differences in object area can reflect true nut size, acquisition geometry, threshold behavior, merged components, fragmentation, or batch effects. The plot alone cannot distinguish these causes.

### Cell 20 — Object warnings

Applies the four object-level structural rules and retains the resulting warning table for consolidation.

### Cells 21–23 — Representative images and segmentation overlays

A NumPy generator seeded with `RANDOM_STATE` samples up to `N_EXAMPLE_IMAGES_PER_KIND` images with at least one recorded object from each sample kind. Sampling is without replacement and therefore reproducible for a fixed DataFrame order, NumPy version, and seed.

When enabled, the notebook overlays connected-component labels on each stored `image_ref`. These plots inspect the already stored segmentation; they do not rerun it.

### Cells 24–25 — Optional raw-versus-database comparison

When enabled, loads the entire raw MATLAB file, preprocesses one hard-coded cube, recomputes its maximum reference image, and displays it alongside the stored `image_ref`.

The comparison is visual only. The code does not calculate a numerical difference, assert equality, compare labels or masks, or use `SEGMENTATION_KWARGS`.

### Cells 26–27 — Object grids

Displays up to `N_OBJECTS_IN_GRID` object crops for each representative image. Objects are sorted by decreasing area by the plotting helper and shown with mask overlays.

Each plot call is wrapped in `try/except`; a plotting failure emits a warning and lets the notebook continue. These warnings are printed only and are not added to `qc_flags_df`.

### Cell 28 — Detailed-object selection

Selects:

- the three largest objects in the database;
- the three smallest objects;
- the first available object for each distinct `object_nut_type`.

`dict.fromkeys` removes duplicate IDs while preserving selection order. The class examples are not stratified by batch or sample kind.

### Cell 29 — Detailed object plots

When `RUN_SPECTRAL_QC_PLOTS=True`, `plot_object_view` displays each selected crop and mask plus its mean spectrum and a plus-or-minus-one-standard-deviation band computed from the object's pixels.

Exceptions are printed and ignored. They are not persisted as QC flags.

### Cells 30–32 — Spectral QC table and summaries

For every object, Cell 31 reads `mean_spectrum` and calculates:

- mean, standard deviation, minimum, maximum, and range across bands;
- fraction of non-finite mean-spectrum values;
- object area, pixel count, class, batch, sample kind, source, and split.

All object mean spectra are stacked into `X_mean` with shape `(n_objects, n_bands)`. Cell 32 summarizes the scalar spectral descriptors by sample kind, object label, and batch.

Neither `spectral_qc_df`, `X_mean`, nor `spectral_summary_df` is saved by the current notebook.

This section assumes a nonempty object database and equal-length mean spectra. `numpy.vstack` fails otherwise.

### Cell 33 — Wavelength recovery

Reads the wavelength axis from the first object. If it is missing or empty, subsequent plots use zero-based band indices. The code does not verify that every object has the same wavelength values.

### Cell 34 — Pure-object class spectra

Filters to pure objects and plots mean spectra grouped by `object_nut_type`. Because the recorded database has more than `2 * MAX_OBJECT_SPECTRA_PER_GROUP` pure objects, it samples up to 80 objects per class using the configured seed.

`plot_spectra(..., reducer="mean_std")` displays the mean curve and a plus-or-minus-one-standard-deviation band across object mean spectra in each class. The band describes between-object variation, not uncertainty in the estimated population mean.

### Cells 35–36 — Pure almond and peanut spectra by batch

Stack all pure almond or pure peanut object mean spectra, create labels such as `batch 1`, and display a mean plus-or-minus-standard-deviation curve for each batch. These cells use all available objects and are not limited by `MAX_OBJECT_SPECTRA_PER_GROUP`.

### Cell 37 — Batch 3 comparison

Selects pure objects from batch 3 and compares almond and peanut mean spectra. The notebook comment specifically identifies batch 3 as potentially noisy, but no numerical hypothesis test or automatic batch-3 warning is implemented.

### Cells 38–39 — Simple spectral outlier candidates

The notebook standardizes two scalar features over all objects:

```text
z_spectrum_mean = z-score of the mean reflectance across bands
z_spectrum_std  = z-score of the within-spectrum standard deviation

spectral_outlier_score = |z_spectrum_mean| + |z_spectrum_std|
```

It ranks all objects by this score, keeps the top 20, and plots their complete mean spectra individually.

This is a visual prioritization heuristic, not the project's anomaly-detection model. It pools pure, mixture, and position-reference objects across classes and batches; ignores covariance and detailed curve shape; always returns a top 20 even if no observation is genuinely abnormal; and does not assign a rejection threshold.

The ranked candidates are not saved and are not added to `qc_flags_df`.

### Cells 40–41 — Image-level pixel spectral distributions

For each representative image, the notebook selects pixels for which the stored label is positive, samples up to 2,000 without replacement, and plots the wavelength-wise mean plus-or-minus-standard-deviation band.

Because pixels are sampled directly, larger objects contribute more pixels than smaller objects. The plot characterizes foreground-pixel variation, not an equally weighted distribution over objects.

Failures are printed and ignored, and do not become persisted QC flags.

### Cells 42–43 — Required-field check

Checks for the default required keys in every image and object record. An empty result means all keys are present, not that all values are semantically valid.

### Cell 44 — Object shape consistency

Builds all object-level dimension checks, displays only failures, and reports whether every object passed.

### Cell 45 — Consolidated QC flags

Combines image warnings, object warnings, missing fields, and bad shapes. It saves `qc_flags.parquet` only when at least one row exists.

If the current run has no flags but a `qc_flags.parquet` from an earlier run already exists, the old file is not removed. Consumers should therefore use the final notebook counts or explicitly clear/version outputs to avoid mistaking a stale file for current findings.

### Cell 46 — Modelling summary

Groups objects by split, sample kind, class, and batch, reporting object count and median area. This reveals whether data needed for training, validation, or projection exist under the stored split policy.

The recorded database assigns every object to `projection`, including pure samples. That originates from `FORCED_SPLIT="projection"` in Notebook 00 and should be reviewed before model fitting.

### Cell 47 — Availability dictionary

Lists the available pure almond and peanut batches and counts mixture and position-reference objects. This is displayed but not saved as a standalone artifact.

### Cell 48 — In-memory QC report

Builds a dictionary with database path, database sizes, warning counts, output paths, and availability. It is displayed as a one-row DataFrame but is not written to disk.

### Cell 49 — Completion summary

Prints essential output paths, structural warning counts, and the next notebook name.

## Persisted table schemas

### `image_qc_summary.parquet`

One row per image with the following fields:

| Category | Fields |
| --- | --- |
| Identification | `clean_key`, `image_id` |
| Sample metadata | `sample_kind`, `nut_type`, `batch`, `position_set`, `description`, Boolean sample flags |
| Cube dimensions | `height`, `width`, `n_bands`, `n_pixels_image` |
| Segmentation | `n_objects_recorded`, `n_labels_positive`, `max_label`, `threshold`, `mask_area_pixels`, `mask_area_ratio` |
| Spectral metadata | `has_wavelengths`, `data_mode` |

`n_labels_positive` counts unique label values greater than zero, whereas `max_label` is the largest integer label. These values match for sequential connected-component labels but need not match for an arbitrary label image with gaps.

### `object_qc_summary.parquet`

One row per object with 27 fields:

| Category | Fields |
| --- | --- |
| Identification and provenance | `object_id`, `source_clean_key`, `source_image` |
| Labels and experiment role | `sample_kind`, `image_nut_type`, `object_nut_type`, `batch`, `position_set`, `split`, Boolean sample flags |
| Object indexing | `label_id`, `object_index` |
| Size and spectra | `area_pixels`, `n_pixels`, `n_bands`, `spectra_shape`, `mean_spectrum_length` |
| Geometry | `centroid_row`, `centroid_col`, `bbox`, `bbox_height`, `bbox_width`, `bbox_area` |
| Measurement metadata | `data_mode` |

Complex tuple values such as `spectra_shape` and `bbox` are converted to Parquet-safe strings by the shared export helper when necessary.

### `qc_flags.parquet`

This optional file contains only structural findings combined by `build_qc_flags_table`. It is absent when the current run finds no flags, unless a stale file from an earlier run remains at the path.

It does not include:

- plotted box-plot outliers;
- the top 20 spectral candidates;
- unusually low but positive mask coverage;
- batch or class imbalance;
- failures caught by visualization `try/except` blocks;
- subjective findings from manual figure review.

## Visualizations and how to interpret them

| Visualization | Intended question | Important limitation |
| --- | --- | --- |
| Image counts by sample kind and nut type | Are expected acquisition categories represented? | Counts records, not data quality. |
| Objects per image | Are some images unusually sparse or dense? | No automatic threshold is applied. |
| Object counts by sample kind and label | Is the object database composition plausible? | Mixture objects are intentionally labeled `unknown`. |
| Object-area box plots | Are some batches shifted, fragmented, or merged? | True biological size and acquisition geometry also affect area. |
| Segmentation overlays | Does each label isolate one nut? | Only a sampled subset is shown. |
| Object grids | Are crops and masks visually plausible across objects? | Limited to 25 objects per sampled image. |
| Individual object view | Does the crop, mask, mean spectrum, and pixel variability look plausible? | Selection emphasizes global size extremes and one example per label. |
| Pure spectra by class | Do almond and peanut average curves differ? | Mean plus/minus standard deviation is descriptive, not a confidence interval. |
| Pure spectra by batch | Are batch shifts or noise visible within class? | Visual evidence is not a statistical batch-effect test. |
| Top spectral candidates | Which objects deserve manual inspection first? | Ranking is based on only two global scalar features. |
| Pixel spectral distribution | Is within-image foreground variability plausible? | Pixels, rather than objects, are equally sampled. |

## Reference run recorded in the notebook

The saved outputs describe the database generated by the preceding notebook. These are reference values, not universal pass/fail criteria.

| Metric | Recorded value |
| --- | ---: |
| HDF5 file size | 115.63 MB |
| Images | 48 |
| Objects | 1,262 |
| Cube size per image | `(370, 318, 63)` |
| Mean-spectrum matrix | `(1262, 63)` |
| Mixture images / objects | 20 / 722 |
| Position-reference images / objects | 20 / 146 |
| Pure almond images / objects | 4 / 214 |
| Pure peanut images / objects | 4 / 180 |
| Image warnings | 0 |
| Object warnings | 0 |
| Missing-field records | 0 |
| Bad-shape records | 0 |
| Consolidated structural flags | 0 |
| Maximum mean-spectrum NaN rate by reported group | 0.0 |
| Wavelengths | 63 values, approximately 960.74–1702.00 nm |

The deterministic representative-image selection in the saved run was:

```text
alm1pea2, alm4pea4,
pea2_pos4, pea4_pos5,
peanut2, almond1
```

The recorded position-reference groups deserve manual attention despite the empty structural flag table:

- batches 1–3 each contain 47 objects across five images, with some images containing 15 objects and others only one;
- batch 4 contains only five objects across five images, exactly one per image;
- median mask coverage is approximately 0.0065–0.0072 for batches 1–3 but only about 0.0011 for batch 4;
- median object area in position-reference batch 4 is 127 pixels, compared with 59–69 pixels in batches 1–3.

These differences may reflect acquisition design or merged objects rather than a software error, but the implemented warning rules do not evaluate them.

The top recorded spectral candidate is `alm3pea2_obj026`, with a QC score of approximately `6.61`. This ranking is contextual to the current pooled database and should not be interpreted as a confirmed peanut anomaly or defective observation.

## Recommended review checklist

### Structural checks

- Confirm that the database path and record counts match the intended build.
- Confirm that `n_objects_recorded` equals the number of positive labels for every image.
- Confirm that object pixel counts equal segmented areas.
- Confirm that all summary-spectrum lengths match `n_bands`.
- Confirm that object and source-image spectral dimensions agree.
- Review every missing-field and bad-shape record.
- Verify that the split distribution matches the planned experiment.

### Segmentation checks

- Ensure background pixels are not labeled as nuts.
- Ensure every intended nut is detected.
- Ensure touching nuts are not merged into one connected component.
- Ensure single nuts are not fragmented into several labels.
- Compare object counts and mask coverage across position sets and batches.
- Inspect the smallest and largest objects for fragments and merged components.

### Spectral checks

- Verify that all spectra are finite and use the expected reflectance scale.
- Compare class curves without assuming that visible separation guarantees model performance.
- Inspect batch shifts within almond and peanut separately.
- Review batch 3 as requested by the notebook, but also inspect every other batch.
- Examine the complete curves and source images of top spectral candidates.
- Check whether spectral differences remain after controlling for object size, acquisition batch, and sample kind.

### Reproducibility checks

- Record the HDF5 database version or hash used for QC.
- Keep the random seed and sampling settings with exported figures or reports.
- Restart and run all cells after source-code changes.
- Clear or version old result files before a new run.
- Export important interactive plots if they are needed for review or audit.

## Common modifications

### Increase representative-image coverage

```python
N_EXAMPLE_IMAGES_PER_KIND = 5
```

For guaranteed coverage, replace the sample-kind loop with stratification by sample kind, nut type, batch, and, where relevant, position set.

### Run without heavy-array reconstruction

```python
RECONSTRUCT_HEAVY_OBJECT_ARRAYS = False
RUN_OBJECT_QC_PLOTS = False
RUN_SPECTRAL_QC_PLOTS = False
```

Also adjust the default required object fields if compact storage is intentionally accepted, and note that later spectral summary plots still run unless separately guarded.

### Add scientific image warnings

The current rule set can be extended with project-specific thresholds, for example:

```python
if row["mask_area_ratio"] < min_expected_ratio:
    ...

if not min_expected_objects <= row["n_objects_recorded"] <= max_expected_objects:
    ...
```

Thresholds should be defined by acquisition type and batch where appropriate; a single global count threshold may be misleading.

### Persist spectral candidates

```python
save_parquet(
    spectral_outlier_candidates,
    RESULTS_DIR / "spectral_outlier_candidates.parquet",
)
```

If these candidates become formal QC flags, define an explicit threshold and preserve the score definition, grouping population, and software version.

### Compare raw and stored reference images numerically

After checking that shapes match:

```python
delta = raw_image_ref - db_image_ref

comparison = {
    "max_abs_error": float(np.nanmax(np.abs(delta))),
    "mean_abs_error": float(np.nanmean(np.abs(delta))),
    "allclose": bool(np.allclose(raw_image_ref, db_image_ref, equal_nan=True)),
}
```

This turns the optional visual comparison into a reproducible numerical check.

## Troubleshooting

### `RuntimeError: Could not find project root`

Launch Jupyter from the project root or its direct `notebooks/` directory. Confirm that `src/` exists at the detected root.

### `FileNotFoundError: Database not found`

Run `00_building_database.ipynb`, verify `DB_H5_PATH`, and confirm that the database was saved successfully. The error text inside Notebook 01 currently uses the shorter filename `00_build_database.ipynb`.

### HDF5 format or count validation fails

Confirm that the file was created by `save_nir_uco_h5` and was not partially written or manually altered. Rebuild it from the raw source after preserving the failed artifact for diagnosis.

### Object plots report missing crop fields

Set `RECONSTRUCT_HEAVY_OBJECT_ARRAYS=True` or build the HDF5 file with `INCLUDE_HEAVY_OBJECT_ARRAYS=True` in Notebook 00.

### `numpy.vstack` fails during spectral QC

Check that `object_db` is nonempty and every `mean_spectrum` is one-dimensional with the same length. Run or adapt the required-field and shape checks earlier if malformed databases must be diagnosed gracefully.

### No wavelength axis is found

Plots fall back to band indices. Verify that `wavelengths` was stored in Notebook 00 and that every object uses the intended axis.

### Spectral plots are slow or unreadable

Reduce sampling limits, inspect fewer representative images, or use summary reducers. Note that Cells 35–37 currently use all matching objects.

### A plotting error does not appear in `qc_flags.parquet`

Several visualization cells catch exceptions and print warnings only. Add explicit flag records if plotting failures must be audited.

### No `qc_flags.parquet` is produced

This is expected when `qc_flags_df` is empty. Verify the final printed count. If a file with that name already exists, check that it is not left over from an earlier flagged run.

### Parquet export fails

Install `pyarrow` or another pandas-compatible engine and confirm that `results/01_quality_check/` is writable.

### Memory use is high

Loading all image cubes already requires substantial memory. Heavy-array reconstruction adds global masks and crops for every object. Disable reconstruction for metadata/spectral-only work or implement lazy HDF5 access for larger datasets.

## Known limitations and maintenance notes

- The notebook introduction and missing-file message refer to `00_build_database.ipynb`, while the supplied notebook is `00_building_database.ipynb`.
- `SEGMENTATION_KWARGS` is defined but unused; no segmentation is rerun in this notebook.
- `RUN_SPECTRAL_QC_PLOTS` controls only Cell 29, not the later spectral sections.
- Representative-image sampling covers sample kinds but not necessarily every class, batch, or position set.
- Core QC table builders directly access several required arrays, so severely malformed databases may fail before `check_missing_required_fields` runs.
- Image warnings do not detect extreme but positive object counts or mask-area ratios.
- Object warnings do not validate all values inside `spectra`, `median_spectrum`, or `std_spectrum`.
- Required-field checks test presence only, not validity.
- Wavelength consistency is inferred from the first object and is not verified across all records.
- The spectral score is a pooled two-feature ranking, not a formal outlier test or anomaly-detection model.
- Spectral candidates, spectral summaries, area summaries, modelling summaries, availability, and the final `qc_report` are not persisted.
- Human findings from interactive plots are not captured in a machine-readable review log.
- Plotting exceptions are printed but not incorporated into the canonical flag table.
- A stale `qc_flags.parquet` can remain when a later run produces no flags.
- The raw-versus-database comparison is hard-coded to one image and is visual rather than numerical.
- No output filenames include a database version, hash, timestamp, or parameter tag.
- The notebook does not provide a single Boolean “database accepted” decision, which is appropriate unless explicit scientific acceptance criteria are defined.

## Downstream use

After structural consistency has passed and visual/spectral review is satisfactory, continue with:

```text
02_matrices_preprocessing.ipynb
```

Downstream analysis can use `image_qc_summary.parquet` and `object_qc_summary.parquet` for lightweight diagnostics without loading full hyperspectral cubes. It should treat `qc_flags.parquet` as a structural warning table rather than a comprehensive scientific validation report.
