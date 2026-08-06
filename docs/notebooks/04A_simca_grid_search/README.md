# Notebook 04A — Référence exhaustive SIMCA interne

`notebooks/04A_simca_grid_search.ipynb` réalise les tâches 27 et 28 du
protocole. Il ne réajuste aucun modèle : il réutilise le domaine, les folds,
les seuils et les prédictions OOF de 03B, puis les statuts de domaine et le
verrou spatial de 03C. Seuls les batches 1–2 sont autorisés.

## Contrat

- Chaque `domain_config_id` reçoit exactement une ligne dans l’audit
  technique, qu’il soit calculable ou en erreur.
- Les seeds sont des répétitions d’une même `calibration_id`, pas des modèles
  concurrents ni des doublons.
- Les seuils 2-way/3-way sont appliqués tels quels. Aucun seuil n’est suggéré,
  recalibré ou sélectionné dans 04A.
- Le post-traitement spatial est celui verrouillé par 03C. La couche incertaine
  reste inchangée.
- Calculabilité, acceptabilité, éligibilité de domaine, équivalence et Pareto
  sont cinq statuts distincts.
- Les configurations calculables qui échouent aux contraintes restent dans
  l’audit.
- Les tracks non soutenus ne sont jamais supprimés silencieusement : chacun
  des huit tracks possède une ligne de synthèse explicite.
- Le Pareto est calculé séparément dans chaque `evaluation_track`, à partir des
  objectifs centralisés dans `SIMCA_EVALUATION_TRACK_SPECS`. Aucun score
  pondéré, plafond, quota ou filtre de diversité n’est utilisé.
- La déduplication exacte choisit le représentant par ordre lexical, jamais par
  performance. Une équivalence de sorties n’est destructive que si les
  vecteurs OOF de scores et de décisions sont tous deux identiques.

## Sorties

- `grid_configurations.parquet` : domaine principal dédupliqué par
  `calibration_id`, avec les seeds et identifiants sources conservés.
- `grid_fold_metrics.parquet` : métriques par seed, fold et image.
- `grid_threshold_metrics.parquet` : agrégats, comptes, dispersion,
  acceptabilité et éligibilité par `calibration_id`.
- `grid_pareto_reference.parquet` : fronts diagnostique et protocolaire dans
  les huit tracks.
- `technical_audit.parquet` : audit exhaustif par `domain_config_id`.
- `duplicate_groups.parquet` : groupes exacts et vecteurs de prédiction
  identiques.
- `calculable_not_acceptable.parquet` : configurations calculables hors
  contraintes.
- `grid_protocol.json` : provenance et empreintes SHA-256 des entrées/sorties.

Les prédictions OOF complètes ne sont pas recopiées dans 04A ; seules les
métriques compactes et les signatures nécessaires à l’audit sont persistées.
