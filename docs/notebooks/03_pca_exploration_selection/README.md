# Notebook 03 — PCA, stabilité et shortlist

## Rôle

Le notebook implémente les tâches 15–16 du protocole. Il ajuste les PCA
uniquement sur les objets QC acceptés des batches 1–2. Les batches 3–4 ne sont
ni ajustés, ni projetés, ni évalués dans cette étape.

## Contrat d’entrée

Le plan de candidats est construit par `build_pca_candidate_plan(...)` à partir
des trois tables du notebook 02 :

- `matrix_summary.parquet` ;
- `m_feasibility.parquet` ;
- `preprocessing_validation.parquet`.

Seules les lignes `accepted` sont admises. Le plan contient `object_mean`,
`object_median`, `all_pixels` sans `m`, et `balanced_pixels` pour les stratégies
`random` et `center` avec `m=10` et `m=20`. Chaque candidat possède un
`candidate_id`, un `training_matrix_id` et l’identifiant verrouillé de l’axe
spectral.

## Stabilité

Les folds sont construits une seule fois au niveau `source_image` et conservent
les deux classes et les batches 1–2 dans les parties apprentissage et
validation. Ils ne varient pas avec les graines.

La stabilité est calculée séparément pour chaque candidat :

- stabilité principale entre folds groupés ;
- stabilité secondaire entre graines, uniquement quand la graine modifie
  l’échantillonnage aléatoire des pixels ;
- bootstrap secondaire au niveau `source_image`.

Les angles principaux/SVD comparent d’abord les sous-espaces, de manière
invariante au signe et à la permutation des composantes. Les corrélations de
composantes restent un diagnostic secondaire.

## Revue visuelle et sélection

`pca_visual_review.pdf` contient une page à six panneaux pour chaque candidat
techniquement valide, avant toute sélection. `pca_artifact_review.parquet`
référence le chemin réel du PDF, la page, son SHA-256 et le `run_fingerprint`.
Une décision `accept`, `warning` ou `reject` documentée est obligatoire ; une
ligne `pending`, une empreinte périmée ou un artefact critique non rejeté bloque
le notebook.

Le `run_fingerprint` utilisé par la revue est toujours celui du run courant : il
n'est jamais recopié manuellement dans la cellule de décisions. Une revue
antérieure peut être rattachée à un nouveau verrou de protocole uniquement si
le PDF régénéré possède exactement le SHA-256 du PDF effectivement relu et si
les `candidate_id` documentés existent tous dans le run courant. Toute
modification d'un seul octet du PDF remet la revue en état bloquant et impose
une nouvelle lecture. Les groupes de décisions sont appliqués par la fonction
vectorisée `apply_pca_artifact_review_decisions(...)`.

Après la revue, la décision est agrégée au niveau
`(matrix_family, preprocessing)`. Un prétraitement doit couvrir toutes les
variantes attendues de sa famille sans aucun candidat `reject` ou techniquement
bloqué. Les métriques Pareto sont agrégées en pire cas entre variantes : minimum
pour un objectif à maximiser, maximum pour un objectif à minimiser.

Le Pareto utilise uniquement la séparation de classes, l’effet batch,
l’instabilité et `ncomp_95`. Les métriques de projection restent disponibles
comme diagnostics, sans intervenir dans la sélection. Tous les prétraitements
non dominés sont conservés ; il n’y a ni filtre par quantile, ni score pondéré,
ni crowding, ni filtre de diversité. `MAX_PCA_PREPROCESSINGS_PER_FAMILY` vaut
`None`, donc aucun plafond n’est actif.

## Sorties

Le dossier PCA contient exactement cinq tables et le PDF de revue :

- `pca_summary.parquet` ;
- `pca_scoring_diagnostics.parquet` ;
- `pca_preprocessing_summary.parquet` ;
- `pca_selected_preprocessings.parquet` ;
- `pca_artifact_review.parquet` ;
- `pca_visual_review.pdf`.

La shortlist porte un `shortlist_id` unique, le hash du protocole, l’empreinte
des entrées et le hash de la revue. Le notebook 03B vérifie ces quatre verrous
avant de construire sa grille.

## Tests

`tests/test_pca_protocol_tasks15_16.py` couvre l’univers de candidats, les
folds, l’invariance des sous-espaces, la revue bloquante, le Pareto complet sans
plafond actif et les verrous lus par 03B. `tests/test_pca_selection.py` vérifie
la couverture stricte et l’agrégation en pire cas.
