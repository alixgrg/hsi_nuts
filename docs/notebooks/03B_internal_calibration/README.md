# Notebook 03B — Calibration interne SIMCA à huit tracks

`notebooks/03B_internal_calibration.ipynb` applique les tâches 17 à 24 du
protocole `8tracks_v3`. Les batches 1–2 sont les seules données de calibration.
Les batches 3–4 sont bloqués et les objets doivent appartenir au rôle
`calibration` du manifeste QC.

## Contrat scientifique

Le notebook matérialise d’abord `track_contracts.parquet`, soit exactement les
tracks E1–E8. Trois identités évitent toute ambiguïté :

- `fit_config_id` identifie le fit train-only, indépendamment de la projection
  et de la décision ;
- `projection_config_id` ajoute le niveau, la méthode de projection et la règle
  SIMCA ;
- `evaluation_config_id` ajoute le track et le mode 2-way/3-way.

Un fit est réutilisé pour les projections autorisées : méthode objet assortie à
l’entraînement pour E1/E2, `all_pixels` pour E3/E4/E7/E8, et comparaison
`object_mean`/`object_median` pour E5/E6.

Le score canonique est la marge :

```text
normalized_ratio = rule_statistic / rule_limit
simca_margin = 1 - normalized_ratio
direct_2way_decision = simca_margin >= 0
```

Les seuils 0,75 et 0,80 ne s’appliquent qu’à l’agrégation secondaire des votes
pixels en objets pour E3/E7. Ils ne modifient jamais une projection objet
directe.

## Folds et absence de fuite

Les folds sont groupés par image et réutilisés à l’identique dans les huit
tracks. Avec quatre images, les partitions possibles sont énumérées de manière
déterministe. L’algorithme minimise lexicographiquement l’incomplétude
classe/batch, le déséquilibre du nombre d’objets, des classes de taille puis de
la taille médiane.

À chaque fold, matrice d’entraînement, prétraitement, PCA/SIMCA et limites sont
ajustés sur le train uniquement. Les limites non finies ou négatives sont
bloquantes. `m` est limité à `(10, 20)` ; avec `under_m_policy="exclude"`, un
objet sous `m` est exclu de la matrice d’entraînement concernée mais reste dans
la projection OOF.

Les exclusions QC amont restent définies au niveau objet. La validité qui
dépend du prétraitement est donc appliquée au plus près de la projection par le
préprocesseur partagé. Une ligne pixel comportant une réflectance `<= 0` est
exclue pour une chaîne `absorbance` stricte ; elle n’est jamais clippée vers une
absorbance extrême. Les nombres de lignes initiales, conservées et exclues sont
enregistrés dans `calibration_audit.parquet` sous
`audit_type=projection_input_filter`. Une projection encore invalide est
journalisée dans les diagnostics de règle sans interrompre les autres fits.

## Métriques et seuils

Les pixels des références pures héritent du label objet : ce sont des vérités
faiblement supervisées. 03B calcule des diagnostics micro-pixel, macro-objet et
macro-image, mais n’utilise ni `small_fragment_recall` ni `fragment_precision`.
Ces métriques nécessitent les annotations indépendantes du batch 4.

Les seuils 3-way directs satisfont toujours `t_bas < 0 < t_haut`. Ils sont
calibrés de façon croisée : pour chaque fold d’évaluation, les couples sont
construits et filtrés sur les OOF des autres folds, puis évalués sur le fold
tenu à l’écart. Un couple final est ensuite recalibré sur l’ensemble des OOF et
verrouillé. Il n’existe aucun fallback hors contraintes et aucun score pondéré.

La réduction du domaine suit l’ordre prescrit : contraintes techniques et de
risque, plus petit `k` sur plateau par track, plus petit `m` sur plateau pour
`balanced_pixels`, consensus des graines, puis Pareto par `evaluation_track`.
Toutes les configurations non dominées sont conservées.

Un track techniquement valide mais sans configuration satisfaisant les
contraintes de risque peut être déclaré explicitement non supporté. Cette
déclaration ne relâche aucune contrainte et ne crée aucun fallback : le track
reste documenté dans `calibration_audit.parquet` avec
`track_status=unsupported` et `failure_reason=risk_constraints`, mais il est
absent de `calibrated_hyperparameters.parquet` et de
`calibration_domain.parquet`. Dans l’exécution actuelle, cette exception est
limitée à E3 (`object_train__pixel_projection__2way`). Tout autre échec reste
bloquant.

## Sorties canoniques

03B écrit une table par type, sans CSV ni graphique par défaut :

- `track_contracts.parquet`
- `internal_calibration_folds.parquet`
- `fold_diagnostics.parquet`
- `fit_diagnostics.parquet`
- `rule_diagnostics.parquet`
- `oof_object_predictions.parquet`
- `oof_pixel_predictions.parquet`
- `projection_shift.parquet`
- `oof_2way_metrics.parquet`
- `pixel_to_object_thresholds_2way.parquet`
- `thresholds_3way.parquet`
- `thresholds_3way_study.parquet`
- `calibrated_hyperparameters.parquet`
- `calibration_audit.parquet`
- `calibration_domain.parquet`
- `checkpoint_manifest.json`

Le nouveau répertoire commence par
`03B_internal_calibration_8tracks_v3_`. Les anciens Parquets exploratoires ne
sont donc jamais mélangés au nouveau schéma.

`calibration_domain.parquet` est l’unique domaine consommé par 04A et 04B. Ces
notebooks vérifient son SHA-256 dans `checkpoint_manifest.json` et ne
reconstruisent ni seuil ni domaine.

Après une interruption du kernel, les sections 8 et 9 peuvent reprendre les
tables compactes déjà écrites et le run de checkpoint complet. Il n’est donc
pas nécessaire de réexécuter la section 5 si ses marqueurs couvrent déjà tous
les `fit_config_id` attendus. Le résolveur ignore les répertoires de run
compatibles mais incomplets.

03C enrichit ce contrat avant 04A/04B : les sorties OOF contiennent les deux
premiers scores PCA et `projection_shift.parquet` conserve les références
train (moyenne et écart-type) nécessaires aux diagnostics stratifiés. Le
contrat de checkpoint est versionné ; un ancien run 03B doit donc être relancé.

## Fonctions principales

- `build_simca_track_contracts`
- `build_reference_object_table`
- `build_calibration_folds`
- `build_internal_calibration_configurations`
- `expand_projection_configurations`
- `run_internal_calibration_8tracks`
- `evaluate_internal_2way_tracks`
- `evaluate_crossfitted_three_way_thresholds`
- `build_internal_calibrated_hyperparameters_8tracks`
- `build_calibration_domain_8tracks`
