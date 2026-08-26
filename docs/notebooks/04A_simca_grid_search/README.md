# Notebook 04A — Audit de référence des modèles SIMCA sélectionnés

`notebooks/04A_simca_grid_search.ipynb` réalise les tâches 27 et 28 sans
ouvrir une seconde sélection. Il audite exclusivement les modèles, runs,
seuils et métriques de validation croisée verrouillés par 03B, puis associe
le statut d'éligibilité de domaine produit par 03C.

## Contrat

- 03B reste l'unique autorité de sélection. 04A ne calcule aucun nouveau
  Pareto, ne modifie pas `selected_models` et ne propose aucun seuil.
- La clé d'exécution reste `(model_id, random_state)`. La seule clé ajoutée aux
  métriques est le couple scientifique `decision_scope, fold_id` ; aucun ID
  synthétique n'est créé.
- Le gros fichier `threshold_metrics.parquet` est filtré en flux par Arrow.
  Seules les politiques effectivement sélectionnées sont converties et
  agrégées.
- Les métriques modèles sont reconstruites avec les mêmes fonctions que 03B,
  puis comparées à `model_metrics.parquet` à la précision de sérialisation
  `float32` centralisée dans `experiment_config`.
- Les huit tracks sont conservés. Un track non soutenu par 03C est marqué
  `diagnostic_only`, jamais éliminé silencieusement.
- Le verrou spatial 03C est vérifié par son hash mais n'est pas réappliqué :
  04A n'évalue aucune nouvelle carte et laisse cette opération au workflow
  aval concerné.

## Sorties

- `selected_model_reference.parquet` : une ligne par `model_id`, avec le track,
  les cardinalités, le statut 03C et l'écart maximal de reproduction.
- `selected_run_fold_metrics.parquet` : une ligne par
  `(model_id, random_state, decision_scope, fold_id)`.
- `audit_manifest.json` : hashes, cardinalités et déclaration explicite de
  non-refit/non-resélection.

Les seuils, hyperparamètres, prédictions OOF et métriques modèles ne sont pas
recopiés : ils restent dans les artefacts 03B faisant autorité et sont reliés
aux sorties 04A par leurs empreintes SHA-256.
