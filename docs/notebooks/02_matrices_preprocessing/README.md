# 02_matrices_preprocessing.ipynb

## Rôle et ordre scientifique

1. créer le manifeste calibration/validation/test/mélange ;
2. vérifier et verrouiller l'axe spectral avec un SHA-256 ;
3. étudier `m=(10, 20, 40, 60, 80, 100)` sans traitement silencieux des
   petits objets ;
4. construire les matrices de calibration et validation ;
5. vérifier finitude, classes, traçabilité et couverture ;
6. ajuster les prétraitements sur la calibration puis transformer la
   validation ;
7. tester la grille SG exacte `(5, 7, 9, 11, 13, 21)`, degré 2.

Le batch 4 est inventorié dans le manifeste, mais aucune matrice de test n'est
construite à cette étape.

## Matrices

- `object_mean`
- `object_median`
- `balanced_pixels` avec stratégies `random` et `center`
- `all_pixels`

Les métadonnées ligne sont réduites à `object_id`, `source_image`, `batch`,
`label`, `sample_kind`, plus les coordonnées pour les matrices pixels.

## Sorties

- `wavelength_config.parquet`
- `m_feasibility.parquet`
- `pixel_sampling_diagnostics.parquet`
- `matrix_summary.parquet`
- `matrix_coverage.parquet`
- `matrix_errors.parquet`
- `preprocessing_validation.parquet`
- `preprocessing_errors.parquet`

Toutes sont enregistrées dans `results/02_matrices_<RESULTS_TAG>/` avec des
schémas compacts définis dans `src/experiment_config.py`.

## Modules principaux

- `src.matrices.matrix_registry`
- `src.matrices.redim_matrix`
- `src.spectra.preprocessing`
- `src.spectra.preprocessing_configs`
- `src.workflows.matrix_preprocessing`
