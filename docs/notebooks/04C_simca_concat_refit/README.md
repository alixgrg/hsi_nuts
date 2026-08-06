# Notebook 04C — validation batch 3 et reconstruction spatiale

`notebooks/04C_simca_concat_refit.ipynb` réalise les tâches 31 à 33 du
protocole. Il refitte les configurations gelées sur les batches 1–2, projette
uniquement le batch 3 et applique les seuils appris par 03B sans les modifier.
Le batch 4 n'est jamais chargé.

## Pool sans score

Le pool vient exclusivement de la référence exhaustive 04A : front Pareto
protocolaire pour les tracks soutenus par 03C et front Pareto diagnostique pour
les tracks `unsupported_domain_shift`. Optuna est seulement une annotation de
provenance. Il n'ajoute, ne retire et ne classe aucun candidat.

Une `calibration_id` multi-seed est développée en plusieurs
`validation_candidate_id`. Les paramètres scientifiques sont comparés
strictement à 03B ; les `random_state`, `fit_config_id` et
`projection_config_id` propres à chaque seed sont conservés. Aucun plafond,
quota, score composite ou filtre de diversité n'est appliqué.

## Refit et projections

Le calcul est factorisé par configuration de données, fit et projection. Les
prédictions continues sont stockées une seule fois par
`projection_config_id`, puis les décisions 2-way/3-way sont appliquées de
façon vectorisée pour chaque candidat. Les checkpoints sont liés par hash au
plan de validation, au pool, au plan d'ablation 04B et au verrou spatial 03C.

Les métriques sont calculées globalement, par seed et par image. Les tracks
pixel utilisent aussi les agrégats macro image et macro objet. Les signatures
de scores et de décisions servent à identifier les équivalences après refit,
mais aucune ligne équivalente n'est supprimée.

## Vérité et post-traitement spatial

Les images du batch 3 sont pures. Dans le ROI segmenté, chaque pixel d'une
image `peanut` est donc une cible exacte et chaque pixel d'une image `almond`
une non-cible exacte. L'arrière-plan reste hors vérité. Les labels de
segmentation sont utilisés comme identifiants de composantes vraies afin que
deux objets adjacents ne soient pas fusionnés par une simple binarisation.

Le manifeste spatial conserve les couches valide/arrière-plan, cible brute,
incertitude, cible post-traitée et vérité. L'incertitude est immuable. La marge
continue reste dans la table de prédictions pixel. Les composantes sont
associées par IoU et documentent tailles, centroïdes, split, merge et rappel
par classe d'aire.

## Garde-fous

Les contraintes centralisées dans
`SIMCA_CONCAT_REFIT_GUARDRAIL_LIMITS` sont appliquées aux valeurs ponctuelles
préspécifiées. Le scope global utilise les limites globales du profil de
risque actif, tandis que `worst_image` utilise ses limites par image. Comptes,
intervalles et pire image sont conservés. Les statuts possibles incluent `pass`,
`calculable_but_not_acceptable`, `technical_failure`,
`unsupported_domain_shift_diagnostic` et
`not_evaluable_no_calibrated_candidate`.

Comme les images pures sont monoclasse, la balanced accuracy n'est pas définie
dans une image isolée. Au niveau macro-image, elle est reconstruite à partir du
taux de raté sur les images cibles et du taux de fausse acceptation sur les
images non-cibles. Le scope `worst_image` contrôle uniquement les risques
conditionnels applicables et l'incertitude 3-way avec les plafonds par image.
La couverture reste reportée et sa complémentarité exacte avec l'incertitude
est vérifiée, mais elle ne constitue pas un second garde-fou bloquant
redondant. Une métrique requise non finie est une erreur technique et non une
violation scientifique. Cette règle est centralisée, versionnée et hashée
séparément du plan de refit ; aucun seuil numérique n'est modifié.

Aucun seuil numérique de rappel des plus petits fragments n'étant gelé avant
le batch 3, cette métrique reste diagnostique. Le notebook ne choisit pas de
seuil a posteriori.

## Sorties

- `validation_object_predictions.parquet` ;
- `validation_pixel_predictions.parquet` ;
- `validation_metrics.parquet` ;
- `pixel_maps_manifest.parquet` ;
- `spatial_components.parquet` ;
- `spatial_component_metrics.parquet` ;
- `validation_guardrails.parquet` ;
- `validation_protocol.json`.

Les couches booléennes du manifeste sont encodées avec
`packbits_zlib_v1`. La fonction `decode_boolean_map` permet de les relire pour
l'audit ou la visualisation.
