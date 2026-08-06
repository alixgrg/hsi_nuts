# 00_building_database.ipynb

## Rôle

Construire la base canonique sans appliquer SNV, MSC ou Savitzky-Golay.
Le notebook vérifie les cubes bruts, parse les métadonnées, retire les bandes
initiales bruitées, segmente les objets, extrait leurs spectres et valide le
roundtrip HDF5 en profondeur.

Toute la configuration vient de `src/experiment_config.py`. Le notebook ne
force plus le rôle `projection`; les rôles scientifiques sont attribués au
notebook 02.

## Contrôles bloquants

- fichier MATLAB présent et cubes 3D numériques ;
- aucun NaN/infini avant ou après le prétraitement bas niveau ;
- métadonnées reconnues avec statut explicite ;
- cohérence cube, masque, labels et axe spectral ;
- diagnostic `too_small`, bordure, proximité et fusion possible ;
- comparaison mémoire/HDF5 des identifiants, champs, dimensions, spectres et
  longueurs d'onde.

## Sorties

- `HSI Data/processed/nir_uco_database.h5`
- `results/00_database/raw_image_manifest.parquet`
- `results/00_database/metadata_parsing_errors.parquet`
- `results/00_database/image_summary.parquet`
- `results/00_database/object_summary.parquet`
- `results/00_database/segmentation_diagnostics.parquet`
- `results/00_database/database_manifest.parquet`

Les Parquet utilisent les schémas compacts définis dans
`src/experiment_config.py`. Les tableaux détaillés restent en mémoire seulement
pendant les contrôles.

## Modules principaux

- `src.data.database`
- `src.io.database_h5`
- `src.workflows.quality_check`

Après toute modification de segmentation, relancer les notebooks 00, 01 et 02.
