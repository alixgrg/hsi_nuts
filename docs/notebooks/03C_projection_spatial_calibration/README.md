# Notebook 03C — audit de projection et verrou spatial

Le notebook `03C_projection_spatial_calibration.ipynb` applique les tâches 25
et 26 aux modèles sélectionnés par 03B, avant toute recherche ou validation
aval.

## Entrées normalisées

03C vérifie puis consomme huit artefacts du manifeste 03B :

- `track_contracts.parquet` ;
- `model_catalog.parquet` ;
- `selected_models.parquet` ;
- `selected_runs.parquet` ;
- `selected_thresholds.parquet` ;
- les prédictions OOF objet et pixel ;
- `projection_shift.parquet`.

`selected_thresholds.parquet` contient aussi des politiques intermédiaires.
03C les filtre explicitement par la clé naturelle sélectionnée
`(model_id, random_state)`, contrôle les portées `direct` et
`pixel_to_object`, puis refuse toute cardinalité ambiguë. Le `projection_id`
sert seulement à joindre les prédictions et les références en mémoire. Il
n'est pas recopié dans les sorties.

## Audit train → projection

Les diagnostics comparent les deux premiers scores PCA, T² (`H`), `Q`, la
limite de règle, le ratio normalisé et la marge SIMCA. Ils sont agrégés de
façon vectorisée pour l'ensemble des observations, les folds, les classes de
taille, les zones bord/cœur, la classe vraie et l'image source.

Les strates d'éligibilité `overall` et `fold` comparent exclusivement les
projections de la classe cible aux références train de cette même classe. Les
non-cibles restent présentes dans les strates descriptives. Chaque track E1–E8
reçoit exactement un statut déclaré dans `src/experiment_config.py`. Un track
sans modèle sélectionné reçoit explicitement
`unsupported_internal_calibration`; aucun modèle ou identifiant n'est inventé.

## Calibration spatiale

La calibration spatiale utilise seulement les exécutions à projection pixel
dont le statut est `eligible` ou `eligible_with_warning`. Les décisions brutes
sont reconstruites avec les seuils directs sélectionnés de 03B. La vérité est
exacte sur les images pures des batches 1–2 : l'intérieur segmenté d'une image
`almond` est non-cible et celui d'une image `peanut` est cible. Toute donnée des
batches 3–4 provoque une erreur.

La clé d'une carte est `(model_id, random_state, source_image)`. La couche
incertaine 3-way est conservée à l'identique et exclue des pixels scorés. Les
candidats spatiaux couvrent exactement les mêmes exécutions. La sélection
équilibre d'abord les exécutions dans chaque `track_id`, puis les tracks entre
eux, avant le départage lexicographique et le choix de complexité minimale.

`spatial_candidate_id` est le seul nouvel identifiant : il est nécessaire pour
relier une combinaison de paramètres spatiaux aux métriques et au verrou. Les
sorties ne contiennent ni identifiant de fit, ni identifiant de projection, ni
identifiant de domaine ou de run supplémentaire.

## Sorties

- `projection_shift_diagnostics.parquet` ;
- `projection_eligibility.parquet` ;
- `spatial_calibration_metrics.parquet` ;
- `fragment_size_classes.parquet` ;
- `spatial_postprocessing_lock.json` ;
- `audit_manifest.json`.

Le manifeste final relie les hashes des entrées 03B aux hashes, schémas et
cardinalités des sorties 03C. Les tables scientifiques utilisent
`model_id`, `random_state` et `track_id` comme seules clés de traçabilité vers
la sélection des modèles.
