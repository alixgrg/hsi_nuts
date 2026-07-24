# Notebook 07 - SIMCA Mixture Application

## Role In The Workflow

`notebooks/07_simca_mixture_application.ipynb` applies the final SIMCA models selected in notebook 06B to mixture images.

This notebook is an application step, not a model-selection step. It does not tune thresholds or rank final models again. It restores the full model configurations selected in 06B, refits each model on pure target batches 1-4, projects mixture objects, computes compact diagnostics, and saves outputs for a later reporting script.

## Execution Order

Run this notebook after:

1. `04C_simca_concat_refit.ipynb`
2. `06A_simca_pure_test.ipynb`
3. `06B_simca_final_selection.ipynb`

Required inputs:

```text
HSI Data/processed/nir_uco_database.h5
results/03_pca_<RESULTS_TAG>/pca_selected_preprocessings.parquet
results/04C_simca_concat_refit_<RESULTS_TAG>/validation_3way_selected_thresholds.parquet
results/06A_simca_pure_test_<RESULTS_TAG>/pure_test_candidate_panel.parquet
results/06B_simca_final_selection_<RESULTS_TAG>/final_selected_models.parquet
```

Outputs are written to:

```text
results/07_simca_mixture_application_<RESULTS_TAG>/
```

## Main Outputs

- `mixture_selected_configs.parquet`: full refit configurations restored from 06B and 06A.
- `mixture_metrics_long.parquet`: compact mixture metrics for the final assigned tracks.
- `mixture_summary.parquet`: small summary table by track and metric level.
- `mixture_object_diagnostics_by_image.parquet`: object-level diagnostics by mixture image.
- `mixture_pixel_diagnostics_by_image.parquet`: pixel-level diagnostics by mixture image.
- `mixture_3way_object_diagnostics_by_image.parquet`: fixed-threshold 3-way object diagnostics by image.
- `mixture_objects.parquet`: combined object-level predictions when enabled.
- `mixture_pixels.parquet`: combined pixel-level predictions when enabled.
- `mixture_3way_objects.parquet`: object-level 3-way decisions when enabled.
- `mixture_guardrails.parquet`: input and protocol checks.
- `mixture_protocol.parquet`: settings and output row counts.

## Logic

1. Load the HDF5 database and selected wavelength configuration.
2. Load PCA-selected preprocessing families.
3. Load the compact final 06B model list.
4. Restore full model parameters from the 06A candidate panel.
5. Attach fixed 3-way thresholds selected before pure test in 04C.
6. Validate that final train batches are pure batches 1-4 and projection filters select mixture objects.
7. Refit selected models on pure target batches 1-4.
8. Project mixture images and compute object, pixel, and fixed-threshold 3-way diagnostics.
9. Save compact tables for downstream reporting.
10. Display a few notebook-level tables and figures only.

## Associated Modules

From `src.workflows.simca_mixture`:

- `build_mixture_projection_filters(...)`: creates the canonical mixture projection filter.
- `restore_mixture_selected_configs(...)`: joins compact 06B selections with full 06A configs.
- `build_mixture_guardrails(...)`: checks the application protocol before refit/projection.
- `run_mixture_application_batches(...)`: reuses the pure-test refit engine for mixture application.
- `prepare_mixture_outputs(...)`: materializes `metrics_long` and compact summaries.
- `save_mixture_outputs(...)`: saves the standard 07 output set.
- `load_existing_mixture_outputs(...)`: reloads existing outputs without refit.
- `choose_mixture_diagnostic_images(...)`: selects a small image set for notebook diagnostics.

From visualization modules:

- `plot_model_metric_ranking(...)`: compact model metric ranking.
- `plot_per_image_performance(...)`: difficult-image overview.
- `plot_mixture_diagnostic_panel(...)`: one spatial diagnostic panel for a selected model/image.

## Reporting Script Preparation

Notebook 07 deliberately avoids generating many figures. The future reporting script should consume `mixture_selected_configs.parquet`, image diagnostics, object predictions, and pixel predictions, then generate exhaustive plots for the best or most informative images.
