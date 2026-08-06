# 01_database_quality_check.ipynb

## Rôle

Valider la base produite au notebook 00 avant toute création de matrice.
Les décisions possibles sont `accepted`, `warning`, `excluded` et
`corrected_segmentation`.

## Contrôles

- NaN/infini et pixels valides dans chaque cube et chaque objet ;
- cohérence exacte du nombre de bandes et des axes spectraux ;
- dimensions des spectres, positions et statistiques par objet ;
- aire, remplissage de bbox, bordure, distance au voisin et fusion possible ;
- échantillonnage visuel par batch et type d'échantillon ;
- inspection des objets à risque et des spectres atypiques.

Une correction de segmentation active explicitement le drapeau imposant de
relancer 00-02.

## Sorties

- `results/01_quality_check/image_qc_summary.parquet`
- `results/01_quality_check/object_qc_summary.parquet`
- `results/01_quality_check/qc_alerts.parquet`
- `results/01_quality_check/qc_review.parquet`
- `results/01_quality_check/exclusion_manifest.parquet`
- `results/01_quality_check/qc_protocol.parquet`
- `results/01_quality_check/protocol_split_manifest.parquet`
- `results/01_quality_check/split_diagnostics.parquet`
- `results/01_quality_check/qc_visual_review_report.pdf`

Every flag requiring visual review must have a `reviewed` status and an
explicit decision. A `pending` flag blocks the notebook; a global Boolean such
as `qc_flags_df = False` is not a valid closure.

Les colonnes persistées sont limitées aux identifiants, métriques nécessaires
aux graphiques et décisions QC.

## Module principal

`src.workflows.quality_check`
