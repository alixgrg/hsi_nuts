# Notebook 04B — contrôle négatif budgété Optuna

`notebooks/04B_simca_optuna_search.ipynb` évalue, sans modifier la sélection,
ce qu'un échantillonnage TPE catégoriel retrouve dans l'univers des modèles
évaluables de 03B. La sélection scientifique reste exclusivement celle de
`03B_selected_models`; 04A en fournit la référence auditée.

## Rôle scientifique

Optuna ne reçoit ici qu'un paramètre catégoriel, `model_id`. Il ne voit donc
aucune géométrie entre les hyperparamètres et ne peut pas apprendre qu'un
modèle voisin est meilleur. Cette expérience n'est pas une nouvelle
optimisation : c'est un contrôle négatif de couverture sous budget, comparé à
un tirage uniforme avec remise.

04B ne réajuste aucun modèle, ne recalcule aucun seuil et ne crée aucun
candidat downstream. Ses sorties sont interdites comme source de sélection.
Les tracks E3 et E4 non soutenus en projection par 03C restent explicitement
`diagnostic_only`.

## Entrées et identité

Le notebook valide les manifestes et empreintes SHA-256 avant tout calcul. Il
consomme seulement :

- depuis 03B : `track_contracts`, `model_catalog`, `model_metrics`,
  `selected_models` et `selection_audit` ;
- depuis 04A : `selected_model_reference.parquet` et `audit_manifest.json`.

L'univers évaluable contient une ligne par `model_id` et par track, après
vérification que toutes les métriques objectives définies dans
`SIMCA_EVALUATION_TRACK_SPECS` sont finies. L'égalité entre la sélection 03B et
la référence 04A est bloquante.

Aucun `calibration_id`, `domain_config_id`, identifiant d'étude ou hash du plan
n'est répété dans les tables. `trial_number` est uniquement une coordonnée de
séquence à l'intérieur du track, pas un nouvel identifiant scientifique.

## Benchmark de couverture

Chaque track reçoit le budget centralisé
`SIMCA_OPTUNA_N_TRIALS_PER_TRACK`. Le `TPESampler` suggère un `model_id`
existant et les objectifs sont lus dans `model_metrics.parquet`; aucun accès
au H5 ou aux batches externes n'est nécessaire.

Pour `N` modèles évaluables, `K` références sélectionnées et `B` tirages
uniformes avec remise, l'espérance du rappel de chaque référence vaut :

`1 - (1 - 1/N)^B`.

Le notebook rapporte le rappel observé et son écart à cette espérance. Comme
le TPE ne dispose que d'une catégorie opaque, cette comparaison mesure son
absence de gain exploitable pour la sélection plutôt qu'une performance
d'optimisation scientifique.

## Sorties minimales

- `categorical_tpe_sampled_models.parquet` :
  `track_id`, `trial_number`, `model_id`, `is_repeat`,
  `is_selected_reference` ;
- `categorical_tpe_coverage.parquet` : une ligne de couverture et de rappel
  pour chacun des huit tracks ;
- `audit_manifest.json` : provenance, hashes, règle analytique et interdiction
  d'usage pour la sélection.

Les objectifs et paramètres des modèles ne sont pas dupliqués dans ces
sorties : ils restent joignables dans les artefacts 03B à partir de
`model_id`.
