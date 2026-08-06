# 01B_spatial_ground_truth.ipynb

## Rôle

Valider une vérité terrain spatiale indépendante pour la présence de
`peanut`, mesurer l’accord inter-annotateurs, exiger l’adjudication des
désaccords et verrouiller les fichiers avant toute analyse de modèle.

Le notebook ne crée pas les annotations humaines. Elles sont produites avec
`scripts/annotate_spatial_ground_truth.py`, qui ne charge aucune prédiction.
Les deux images sélectionnées (`almond4` et `peanut4`) sont toutes les deux
annotées indépendamment par deux annotateurs.

## Sémantique

- `target_mask=True` : le tissu de peanut occupe au moins 50 % du pixel.
- `target_mask=False` avec `validity_mask=True` : peanut absent.
- `validity_mask=False` dans la ROI : pixel ambigu.
- Hors de `labels > 0` : pixel non évalué, jamais compté comme vrai négatif.

Le contrat complet est stocké dans
`docs/protocol/spatial_annotation_protocol.json` et son hash est enregistré
dans chaque ligne du manifeste.

## Création des quatre annotations

Exécuter séparément :

```powershell
python scripts\annotate_spatial_ground_truth.py --image almond4 --annotator annotator_1
python scripts\annotate_spatial_ground_truth.py --image almond4 --annotator annotator_2
python scripts\annotate_spatial_ground_truth.py --image peanut4 --annotator annotator_1
python scripts\annotate_spatial_ground_truth.py --image peanut4 --annotator annotator_2
```

Dans l’éditeur : `p` marque peanut, `n` marque non-peanut, `a` marque un
pixel ambigu, `u` annule la dernière zone, `s` enregistre et `q` annule.

Des masques créés avec un autre outil peuvent être importés :

```powershell
python scripts\annotate_spatial_ground_truth.py `
  --image peanut4 `
  --annotator annotator_1 `
  --target-mask target.npy `
  --validity-mask validity.npy
```

Les données humaines primaires sont conservées sous
`HSI Data/annotations/spatial_gt_v1/`. Le script produit un masque de ROI, un
masque cible, un masque de validité et un fichier JSON de provenance.

## Sorties dérivées du notebook

- `spatial_ground_truth_manifest.parquet`
- `fragment_reference_components.parquet`
- `annotation_agreement.parquet`
- `annotation_adjudication.parquet`
- `spatial_annotation_protocol.json`
- `spatial_ground_truth_lock.json`

Tout statut `pending`, tout accord manquant, tout désaccord non adjugé ou
toute modification d’un fichier après verrouillage est bloquant.
