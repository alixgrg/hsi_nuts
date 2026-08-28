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

Le front officiel utilise uniquement l'endpoint `direct`, avec une tolérance
Pareto de 0,01. Les objectifs sont resserrés sur les compromis qui doivent être
arbitrés : raté cible et fausse acceptation en 2-way ; ajout de l'incertitude
cible en 3-way ; ajout du raté macro-objet pour les tracks pixel. La projection
`pixel_to_object`, l'incertitude non-cible et la balanced accuracy restent des
diagnostics supporting afin d'éviter qu'une multiplication d'objectifs
corrélés ne rende presque tous les modèles Pareto.

Deux informations sont conservées :

- le front diagnostique, calculé parmi les unités techniquement calculables ;
- le front protocolaire, limité aux domaines soutenus et aux candidats admissibles selon le statut 04C.

Les unités dominées et non soutenues restent dans l'audit avec un motif explicite.

## Tâche 35 — Robustesse multi-seeds

Toutes les unités Pareto faisant appel à `balanced_pixels` avec stratégie `random` sont refittées sur les dix seeds centralisés. Les seuils, prétraitements, PCA, règle SIMCA et paramètres spatiaux restent verrouillés. Les méthodes déterministes ne subissent pas de répétition artificielle.

Le diagnostic conserve moyenne, dispersion, quantiles, étendue, pire seed et
désaccord de décision. Une unité stochastique doit disposer de tous les seeds
prévus. Seule l'instabilité des métriques de raté cible préspécifiées est
bloquante sur les tracks soutenus ; E3/E4 restent diagnostiques. Le désaccord
de décision sur les entités cibles est bloquant, tandis que le désaccord global
est un avertissement. `robust_with_supporting_warnings` reste admissible pour
le registre pré-batch-4, avec ses motifs explicitement conservés.

## Tâche 36 — Ablations preregistrées

Seules les lignes du plan 04B sont évaluées. Les comparaisons opportunistes d'un facteur à la fois ont été supprimées car elles confondent les paramètres. Les contrastes disponibles sont appariés par identifiants gelés ; les variantes de seuil sont recalculées à partir des marges 04C sans refit ; les cartes brute et post-traitée sont comparées à partir des sorties spatiales existantes.

Une ablation impossible, une interaction sans quatre cellules gelées ou une métrique manquante produit une ligne `not_estimable` au lieu d'être supprimée.

## Tâche 37 — Revue pré-batch-4

05 ne refait pas un choix lexicographique après le Pareto. Il conserve dans le
registre de test pur les modèles du front protocolaire qui passent les
garde-fous bloquants 04C, couvrent les seeds requis et ne présentent pas
d'instabilité bloquante. Les avertissements 04C et de stabilité sont propagés
sans exclusion. Pour E3/E4, un front diagnostique est calculé et audité, mais
aucun modèle n'entre dans le registre de test pur tant que le statut amont reste
`unsupported_domain_shift`.

## Tâche 38 — Incertitude et contrastes

Les intervalles et tests sur batch 3 sont présentés comme résultats de sélection secondaires. Avec moins de cinq images indépendantes, un intervalle groupé par image est déclaré non estimable ; au moins vingt images seraient requises pour une interprétation inférentielle primaire. Tous les contrastes gelés sont émis, y compris lorsqu'un track ou une métrique manque. Les courbes risque–couverture sont calculées sur les marges continues des modèles 2-way et 3-way, avec poids égal des seeds et agrégation macro image pour les pixels. Elles fournissent les estimations descriptives H4 mais ne peuvent pas modifier les seuils verrouillés.

## Sorties

Le répertoire de sortie est
`results/05_simca_validation_robustness_8tracks_v5_<RESULTS_TAG>/` et contient
notamment :

- `validation_selection_units.parquet`, `validation_selection_members.parquet`,
  `validation_pareto_candidates.parquet` et `validation_pareto_audit.parquet` ;
- `robustness_seed_executions.parquet`, `robustness_seed_metrics.parquet`,
  `model_seed_stability.parquet` et `seed_decision_disagreement.parquet` ;
- `robustness_review_guardrails.parquet`, `track_scoring_flags.parquet` et
  `pure_test_candidate_registry.parquet` ;
- `statistical_uncertainty.parquet` et `risk_coverage_curves.parquet` ;
- `robustness_review_protocol.json` et `robustness_review_lock.json`.

Les tables détaillées sont stockées au format long lorsque cela évite une multiplication inutile des colonnes. Les prédictions multi-seeds intermédiaires ne sont pas persistées : seuls les diagnostics nécessaires au protocole sont conservés.

## Configuration centralisée

Les paramètres scientifiques se trouvent dans `src/experiment_config.py`, principalement sous les préfixes :

- `SIMCA_ROBUSTNESS_*` pour Pareto, seeds, stabilité, ablations et risque–couverture ;
- `SIMCA_FINAL_*` pour le profil de risque, les garde-fous spatiaux et la règle lexicographique ;
- `SIMCA_EVALUATION_TRACKS` et `SIMCA_EVALUATION_TRACK_IDS` pour les huit tracks.

Toute modification de ces valeurs doit précéder l'exécution du notebook. Le manifeste final enregistre leur empreinte et verrouille la sélection avant le batch 4.
