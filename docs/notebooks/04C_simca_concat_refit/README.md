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
selon un contrat versionné à deux niveaux. Les limites de sécurité
`blocking` peuvent exclure un candidat ; les limites `warning` documentent un
compromis défavorable sans l'exclure. La décision `direct` est l'endpoint
primaire des tracks objet comme pixel. Pour les tracks pixel, la projection
`pixel_to_object` reste disponible comme diagnostic secondaire, mais ne peut
pas invalider la carte pixel primaire.

Le contrat dur conserve un taux de raté cible global au plus égal à 5 %. Pour
les tracks pixel soutenus, le taux macro-objet de raté cible doit rester au plus
égal à 10 %. Le taux de fausse acceptation déclenche un avertissement au-delà
de 30 % et devient bloquant au-delà de 40 %. En 3-way, la couverture déclenche
un avertissement sous 70 % et devient bloquante sous 60 % ; l'incertitude cible
au-delà de 20 %, l'incertitude non-cible au-delà de 30 % et la balanced
accuracy conditionnelle sous 70 % sont des avertissements.

Le pire taux de raté par image n'est bloquant que si au moins cinq images
indépendantes contribuent à l'estimation. En dessous, sa valeur et son
dépassement sont conservés comme diagnostic non bloquant. Les tracks E3/E4,
non soutenus par le diagnostic de shift 03C, restent entièrement
`diagnostic_only` : leurs métriques et leur front descriptif sont produits,
mais aucun garde-fou 04C ne peut les promouvoir vers le chemin protocolaire ni
les exclure de ce chemin puisqu'ils n'y appartiennent pas.

Ces limites ont été révisées après inspection des résultats du batch 3. Le
manifeste le déclare avec `batch3_used_to_choose_thresholds=True` et
`independent_validation_claim=False`. Elles servent donc à la sélection et à
la description des compromis ; elles ne constituent pas une validation
prospective indépendante. Une estimation non biaisée des performances devra
reposer sur un holdout encore non consulté.

Comme les images pures sont monoclasse, la balanced accuracy n'est pas définie
dans une image isolée. Au niveau macro-image, elle est reconstruite à partir du
taux de raté sur les images cibles et du taux de fausse acceptation sur les
images non-cibles. Le scope `worst_image` contrôle uniquement les risques
conditionnels applicables. Une métrique primaire requise non finie est une
erreur technique et non une violation scientifique. Une métrique secondaire
non évaluable reste diagnostique. Chaque règle possède un `rule_id` stable,
une sévérité et, si nécessaire, un nombre minimal d'unités indépendantes.
Cette règle est centralisée, versionnée et hashée séparément du plan de refit.

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
