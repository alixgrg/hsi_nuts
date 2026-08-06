# Notebook 04B — benchmark budgété Optuna 8-tracks

`notebooks/04B_simca_optuna_search.ipynb` réalise les tâches 29 et 30 du
protocole. Il crée un registre de huit études distinctes E1–E8. Un track sans
domaine calibré conserve une étude vide explicite ; un track non soutenu par
03C reste une étude diagnostique et ne devient jamais candidat downstream.

## Source unique

04B ne recharge pas le H5 et ne réajuste aucun modèle. Il consomme uniquement
les sorties contractuelles de 04A : domaine dédupliqué, métriques exhaustives,
audit technique et fronts Pareto. Les empreintes SHA-256 du manifeste 04A sont
vérifiées avant la création des études. Les seuils, folds et graines restent
ceux de 03B/04A ; les batches 3 et 4 sont inaccessibles à l'objectif.

Optuna suggère un seul paramètre catégoriel : `domain_config_id`. Les graines
de calibration ne sont pas des candidats concurrents. Le `calibration_id`
relie chaque trial aux métriques et à la configuration principale de 04A.

## Objectifs et statuts

Les noms et directions des objectifs sont dérivés de
`SIMCA_EVALUATION_TRACK_SPECS`. Les projections pixel utilisent les métriques
macro image et macro objet définies par 04A. Aucun objectif de stabilité
supplémentaire, signe artificiel ou score pondéré n'est introduit.

Une erreur technique ou un objectif non fini est pruné et documenté. Une
configuration calculable hors garde-fous reste un trial complet : elle peut
appartenir au front diagnostique mais pas au front protocolaire. Calculabilité,
acceptabilité et éligibilité restent trois statuts indépendants.

## Comparaison à l'exhaustif

Le rappel est mesuré contre les flags de `grid_pareto_reference.parquet`, sans
recalcul local du front. Sous `B` tirages uniformes avec remise parmi `N`
configurations, l'espérance du rappel vaut `1 - (1 - 1/N)^B`. Les seuils de
conclusion `useful`, `neutral` ou `insufficient` sont centralisés dans
`experiment_config.py`. Une recherche insuffisante ne retire jamais les
candidats exhaustifs non visités.

## Gel du plan d'ablation

La fin de 04B produit `preregistered_ablation_plan.parquet`. Les références
proviennent du front exhaustif 04A, jamais du sous-ensemble Optuna. Les paires
exactes, sensibilités de seuil, opérations spatiales et interactions autorisées
sont figées avant la prochaine exécution 8-tracks du batch 3. Les sorties
legacy de 04C/05 sont déclarées comme exposition exploratoire antérieure ; le
gel ne constitue pas une revendication de première ouverture historique du
batch 3.

04C vérifie le hash de ce plan avant de charger le batch 3.

## Sorties

- `optuna_trials.parquet` : tous les trials, doublons, états et durées ;
- `optuna_pareto_candidates.parquet` : identifiants et flags Pareto compacts ;
- `optuna_errors.parquet` : échecs techniques ;
- `optuna_search_efficiency.parquet` : une ligne pour chacun des huit tracks ;
- `preregistered_ablation_plan.parquet` : plan prospectif versionné ;
- `optuna_protocol.json` : provenance, hashes, études et règles d'interprétation ;
- `optuna_studies.sqlite3` : stockage reproductible des huit études.

04B ne choisit pas de modèle final et n'applique ni quota ni filtre de
diversité.
