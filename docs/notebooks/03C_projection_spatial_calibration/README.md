# Notebook 03C — changement de domaine et verrou spatial

Le notebook `03C_projection_spatial_calibration.ipynb` applique les tâches 25
et 26 avant toute exécution de 04A ou 04B.

## Entrées

03C consomme exclusivement les artefacts verrouillés de 03B : contrats des
huit tracks, audit de sélection, prédictions OOF objet et pixel, références
train–projection et domaine de calibration. Les observations doivent
appartenir aux batches 1–2. Un checkpoint 03B complet et compatible peut être
réutilisé : il suffit alors de reconstruire les sorties des sections 8 et 9,
sans relancer les fits de la section 5.

## Tâche 25

Les diagnostics comparent les deux premiers scores PCA, T² (`H`), `Q`, la
limite de règle, le ratio normalisé et la marge SIMCA. Ils sont produits pour
l’ensemble des observations, les folds, les classes de taille, les zones
bord/cœur, la classe vraie et l’image source. Les règles d’éligibilité sont
déclarées dans `src/experiment_config.py` et n’utilisent que les strates
globales et par fold, afin que les petites strates descriptives ne déclenchent
pas un rejet instable.

Les strates globales et par fold comparent exclusivement les projections de la
classe cible `peanut` aux références train de la classe cible. Les amandes sont
des non-cibles que SIMCA doit précisément rejeter : les inclure dans le taux de
changement de domaine rendrait mécaniquement un bon classifieur non supporté.
Elles restent disponibles dans les strates descriptives, notamment
`truth_class=non_target`. Les écarts-types petits mais non nuls sont conservés
à leur échelle réelle ; seul un écart-type exactement nul produit un décalage
non borné explicite.

Chaque track reçoit exactement un statut : `eligible`,
`eligible_with_warning`, `unsupported_domain_shift` ou
`unsupported_internal_calibration`. Ce dernier statut matérialise un track
déclaré non supporté par 03B, tel que E3 : aucune projection n’est inventée et
aucun diagnostic spatial n’est exigé pour lui. Un track non soutenu reste dans
les sorties scientifiques ; 04A/04B l’excluent explicitement de la recherche
en affichant son nom.

## Tâche 26

La vérité spatiale est automatique sur les images pures : à l’intérieur du
masque segmenté, tous les pixels `almond*` sont non-cibles et tous les pixels
`peanut*` sont cibles. L’arrière-plan est indisponible. Toute mixture ou image
des batches 3–4 provoque une erreur bloquante.

Le calibrage compare systématiquement les cartes brutes aux variantes de
connectivité, morphologie et aire minimale. La couche incertaine est conservée
à l’identique et exclue des pixels scorés : elle n’est jamais transformée en
non-cible. Le verrou est appris uniquement sur les tracks pixel `eligible` ou
`eligible_with_warning` ; un track `unsupported_domain_shift` ne peut pas
influencer les paramètres utilisés par les tracks exécutables. La sélection
est globale à ce domaine soutenu et équilibre les tracks en deux temps :
moyenne des configurations dans chaque track, puis moyenne à poids égal des
tracks. Elle suit ensuite la priorité lexicographique avec plateau et choisit
la solution la moins complexe. Une moyenne partielle qui ignorerait des
métriques non finies est interdite. Le JSON final contient la provenance, la
pondération, la grille, la version de l’aire minimale et les empreintes des deux
tables de métriques.

Les cartes sont construites séparément par `domain_config_id` et image. Toute
coordonnée dupliquée, marge non finie, décision inconnue, seuil manquant ou
grille dont l’identifiant ne correspond pas aux paramètres provoque une erreur
bloquante avant la sélection.

## Sorties

- `projection_shift_diagnostics.parquet`
- `projection_eligibility.parquet`
- `spatial_calibration_metrics.parquet`
- `fragment_size_classes.parquet`
- `spatial_postprocessing_lock.json`

Ces cinq artefacts sont écrits sous
`results/03C_projection_spatial_calibration_<results_tag>/`.
