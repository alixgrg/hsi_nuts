# Notebook 05 — Robustesse, sélection et verrouillage

## Rôle

`notebooks/05_simca_validation_robustness.ipynb` exécute les tâches 34 à 38 du protocole 8-tracks sur le batch 3. Il transforme les résultats gelés des notebooks précédents en une sélection finale traçable avant l'ouverture du batch 4.

La sélection est volontairement non scalarisée :

- l'unité scientifique est `calibration_id`, jamais le seed ;
- les seeds d'une même calibration ont le même poids ;
- les huit `evaluation_track` sont analysés séparément ;
- aucun score composite, filtre de diversité, quota de famille ou plafond de candidats n'est appliqué ;
- aucun doublon n'est supprimé entre tracks ;
- un track peut se terminer sans modèle acceptable ;
- le batch 4 n'est ni chargé ni utilisé.

## Dépendances amont

Le notebook doit être exécuté après 03B, 03C, 04A, 04B et 04C. Il réutilise notamment :

- le domaine calibré et les hyperparamètres de 03B ;
- les statuts `eligible`, `eligible_with_warning` ou non soutenus et le verrou spatial de 03C ;
- la référence Pareto exhaustive de 04A ;
- les trials Optuna et le plan d'ablation preregistré de 04B ;
- les métriques, décisions continues, garde-fous et métriques spatiales du batch 3 de 04C ;
- le plan de contrastes gelé dans `docs/protocol/planned_contrasts.parquet`.

Les empreintes enregistrées par 04C sont contrôlées avant calcul. Un artefact modifié, un plan d'ablation différent ou un domaine reconstruit différemment provoque une erreur bloquante.

## Tâche 34 — Pareto par track

Les lignes 04C sont agrégées par `evaluation_track` et `calibration_id`. Les seeds sont moyennés à poids égal et restent disponibles dans une table de membres pour l'audit. Les métriques pixel utilisent prioritairement les agrégats macro image/objet et les métriques de fragments.

Deux informations sont conservées :

- le front diagnostique, calculé parmi les unités techniquement calculables ;
- le front protocolaire, limité aux domaines soutenus et aux candidats admissibles selon le statut 04C.

Les unités dominées et non soutenues restent dans l'audit avec un motif explicite.

## Tâche 35 — Robustesse multi-seeds

Toutes les unités Pareto faisant appel à `balanced_pixels` avec stratégie `random` sont refittées sur les dix seeds centralisés. Les seuils, prétraitements, PCA, règle SIMCA et paramètres spatiaux restent verrouillés. Les méthodes déterministes ne subissent pas de répétition artificielle.

Le diagnostic conserve moyenne, dispersion, quantiles, étendue, pire seed et désaccord de décision. Une unité stochastique doit disposer de tous les seeds prévus et satisfaire la politique de sécurité sur chacun d'eux. `robust_with_warning` reste visible mais n'est pas assimilé à `robust` pour le verrou final.

## Tâche 36 — Ablations preregistrées

Seules les lignes du plan 04B sont évaluées. Les comparaisons opportunistes d'un facteur à la fois ont été supprimées car elles confondent les paramètres. Les contrastes disponibles sont appariés par identifiants gelés ; les variantes de seuil sont recalculées à partir des marges 04C sans refit ; les cartes brute et post-traitée sont comparées à partir des sorties spatiales existantes.

Une ablation impossible, une interaction sans quatre cellules gelées ou une métrique manquante produit une ligne `not_estimable` au lieu d'être supprimée.

## Tâche 37 — Politique finale allergène

Le profil `allergen_safety_strict_v1` donne la priorité au risque de rater une peanut. Les garde-fous portent sur le taux de raté cible, la pire image, le pire seed, les faux positifs, l'incertitude 3-way et, pour les tracks pixel, Dice/IoU et les fragments.

Après les contraintes dures, un choix lexicographique déterministe est appliqué dans chaque track : raté cible, pire image, pire seed, rappel des petits fragments, faux positifs, incertitude, stabilité, simplicité du prétraitement, nombre de composantes puis temps de calcul. Les tolérances d'équivalence sont centralisées. Une seule configuration primaire est retenue par track ; les alternatives équivalentes sont verrouillées avec elle.

## Tâche 38 — Incertitude et contrastes

Les intervalles et tests sur batch 3 sont présentés comme résultats de sélection secondaires. Avec moins de cinq images indépendantes, un intervalle groupé par image est déclaré non estimable ; au moins vingt images seraient requises pour une interprétation inférentielle primaire. Tous les contrastes gelés sont émis, y compris lorsqu'un track ou une métrique manque. Les courbes risque–couverture sont calculées sur les marges continues des modèles 2-way et 3-way, avec poids égal des seeds et agrégation macro image pour les pixels. Elles fournissent les estimations descriptives H4 mais ne peuvent pas modifier les seuils verrouillés.

## Sorties

Le répertoire de sortie est `results/05_simca_validation_robustness_8tracks_v3_<RESULTS_TAG>/` et contient :

- `track_pareto_candidates.parquet` et `track_pareto_audit.parquet` ;
- `robustness_diagnostics.parquet` et `robustness_summary.parquet` ;
- `ablation_effects.parquet` et `ablation_interactions.parquet` ;
- `final_safety_guardrails.parquet` ;
- `final_selected_models.parquet` et `locked_models.parquet` ;
- `statistical_uncertainty.parquet`, `planned_contrasts.parquet` et `risk_coverage_curves.parquet` ;
- `final_selection_protocol.json` et `final_lock_manifest.json`.

Les tables détaillées sont stockées au format long lorsque cela évite une multiplication inutile des colonnes. Les prédictions multi-seeds intermédiaires ne sont pas persistées : seuls les diagnostics nécessaires au protocole sont conservés.

## Configuration centralisée

Les paramètres scientifiques se trouvent dans `src/experiment_config.py`, principalement sous les préfixes :

- `SIMCA_ROBUSTNESS_*` pour Pareto, seeds, stabilité, ablations et risque–couverture ;
- `SIMCA_FINAL_*` pour le profil de risque, les garde-fous spatiaux et la règle lexicographique ;
- `SIMCA_EVALUATION_TRACKS` et `SIMCA_EVALUATION_TRACK_IDS` pour les huit tracks.

Toute modification de ces valeurs doit précéder l'exécution du notebook. Le manifeste final enregistre leur empreinte et verrouille la sélection avant le batch 4.
