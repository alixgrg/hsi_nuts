# Project Configuration

`src/experiment_config.py` is the central source of truth for the active
scientific settings. `src/protocol_governance.py` validates these
settings, materializes the eight-track contract and freezes the task-01/task-02
artifacts in `docs/protocol/`.

## Notebook 03C

La configuration des tâches 25–26 est centralisée dans
`src/experiment_config.py` sous les préfixes
`PROJECTION_DOMAIN_*` et `SPATIAL_CALIBRATION_*`. Elle fixe avant les batches
3–4 les seuils d’éligibilité, les dimensions descriptives, la largeur
bord/cœur, la grille morphologique, les classes d’aire, la tolérance de plateau
et la politique de vérité automatique des images pures. Toute modification de
ces valeurs change le verrou du protocole.

For the active pipeline, every experiment choice now lives in this module:
paths, spectral trimming, segmentation, QC policy, protocol batches, matrix
families, the `m` grid, the under-`m` policy, SG parameters, preprocessing
chains, projection levels, decision modes, inference rules and compact output
schemas. Notebook variables are only aliases or paths derived from these
values.

The frozen protocol is identified by:

- `PROTOCOL_VERSION = "8tracks_v3"`
- `RESULTS_SCHEMA_VERSION = "8tracks_v2"`
- `PROTOCOL_STATUS = "frozen"`
- `PROTOCOL_REGISTRATION_MODE = "prospective_amendment_tasks25_26"`

La version `8tracks_v3`, gelée le 3 août 2026, ajoute avant inspection du batch
3 les tâches 25–26 : seuils fixes de changement de domaine, conservation
explicite des tracks non soutenus et verrou spatial global appris uniquement
sur les cartes OOF d’images pures des batches 1–2.

Run the blocking audit with:

```powershell
conda run -n hsi-nuts python scripts\freeze_protocol.py --check-only
```

Regenerate the declared artifacts only when the configuration is unchanged:

```powershell
conda run -n hsi-nuts python scripts\freeze_protocol.py --overwrite
```

Verify the current files against their semantic hashes and checksums with:

```powershell
conda run -n hsi-nuts python scripts\freeze_protocol.py --verify
```

A scientific amendment requires a new `PROTOCOL_VERSION`; it must never be
introduced by overwriting the frozen version silently.

## Canonical Settings For Notebooks 00-02

- `RAW_MAT_RELATIVE_PATH`, `DATABASE_H5_RELATIVE_PATH` and the three result
  directory constants define the project paths.
- `SPECTRAL_START_NM`, `SPECTRAL_END_NM`, `N_BANDS_RAW`,
  `N_REMOVE_START`, `N_STOP_END` and `DATA_MODE` define low-level spectral
  handling.
- `SEGMENTATION_KWARGS` and `SEGMENTATION_MERGE_WARNING_THRESHOLDS` define
  segmentation and its geometry warnings.
- `QC_POLICY` defines exclusions and warning thresholds.
- `PROTOCOL_CALIBRATION_BATCHES`, `PROTOCOL_VALIDATION_BATCHES` and
  `PROTOCOL_TEST_BATCHES` define the scientific roles before matrix creation.
- `BALANCED_SAMPLING_M_VALUES`, `BALANCED_SAMPLING_UNDER_M_POLICY` and
  `BALANCED_SAMPLING_SEEDS` define the technical study of `m`.
- `SG_WINDOW_CHOICES=(5, 7, 9, 11, 13, 21)` and `SG_POLYORDER=2` define the
  exact Savitzky-Golay grid.
- `PREPROCESSING_CONFIGS_TO_COMPARE` contains the complete protocol list.
- `*_OUTPUT_COLUMNS` and `*_REQUIRED_COLUMNS` deliberately keep persisted
  tables compact.

## Shared Experiment Settings

- `DEFAULT_WAVELENGTH_MODE`: spectral namespace used by the current workflow.
- `DEFAULT_RESULTS_TAG`: result-folder suffix used by the active notebooks.
- `TARGET_CLASS`: target nut class for one-class detection.
- `NON_TARGET_LABEL`: non-target class label.
- `REFERENCE_CLASSES`: pure reference classes used in PCA and SIMCA reference workflows.
- `RANDOM_STATE`: default seed for deterministic sampling and scoring diagnostics.

## Batch Protocol

- `SIMCA_TRAIN_BATCHES = (1, 2)`: pure batches used to train validation-stage SIMCA models.
- `SIMCA_VALIDATION_BATCHES = (3,)`: pure validation batch.
- `PURE_TEST_TRAIN_BATCHES = (1, 2, 3)`: pure batches used before external pure-test evaluation.
- `PURE_TEST_BATCHES = (4,)`: held-out pure test batch.
- `MIXTURE_FINAL_TRAIN_BATCHES = (1, 2, 3, 4)`: pure batches available before final mixture application.

Batch 4 must stay out of notebook 03 when it is used as the external pure-test batch in notebook 04C.

## SIMCA Grid Reduction

- `SIMCA_ALPHA_VALUES = [0.01]`
- `SIMCA_OBJECT_THRESHOLDS = [0.75, 0.80]`

These values encode the reduced grid agreed after the audit. Downstream notebooks should import them instead of redefining local grids.

## SIMCA Parent Tracks And Eight Evaluation Tracks

Four `parent_track` remain available to share fitted models and preserve
compatibility with the notebooks under refactoring:

- `object_matrix_2way`
- `object_matrix_3way`
- `pixel_matrix_2way`
- `pixel_matrix_3way`

They live in `SIMCA_PARENT_TRACKS` and `SIMCA_PARENT_TRACK_SPECS`.
`SIMCA_SELECTION_TRACKS` is a temporary backward-compatible alias for these
four parents.

Scientific evaluation, thresholding and Pareto use the projection-aware
`SIMCA_EVALUATION_TRACKS`:

| ID | `evaluation_track` | Primary unit |
|---|---|---|
| E1 | `object_train__object_projection__2way` | object |
| E2 | `object_train__object_projection__3way` | object |
| E3 | `object_train__pixel_projection__2way` | pixel/fragment |
| E4 | `object_train__pixel_projection__3way` | pixel/fragment |
| E5 | `pixel_train__object_projection__2way` | object |
| E6 | `pixel_train__object_projection__3way` | object |
| E7 | `pixel_train__pixel_projection__2way` | pixel/fragment |
| E8 | `pixel_train__pixel_projection__3way` | pixel/fragment |

Every specification records its parent, training family, projection level,
decision mode, primary unit, SIMCA score contract, metrics and Pareto
directions. Pareto is calculated independently by `evaluation_track`.

All direct object and pixel decisions use `simca_margin`. The fixed values
0.75/0.80 occur only as secondary 2-way pixel-to-object aggregation thresholds
in E3 and E7. Object projections E1/E2/E5/E6 never use a pixel-ratio threshold.
The 3-way pixel-to-object policy is calibrated separately on grouped OOF
outputs and does not reuse 0.75/0.80.

SIMCA candidate identity is defined by `SIMCA_CANDIDATE_ID_COLUMNS`. The stable id is created by `src.workflows.simca_candidates.simca_candidate_key(...)` and added to tables with `add_simca_candidate_ids(...)` or `deduplicate_simca_candidates(...)`.

Candidate and evaluation output schemas are documented by:

- `SIMCA_PCA_SHORTLIST_REQUIRED_COLUMNS`
- `SIMCA_CANDIDATE_CONFIG_REQUIRED_COLUMNS`
- `SIMCA_CANDIDATE_EVALUATION_REQUIRED_COLUMNS`
- `SIMCA_FINAL_MODEL_SELECTION_REQUIRED_COLUMNS`

The PCA shortlist from notebook 03 must remain scoped by matrix family. Use `build_pca_preprocessing_configs_by_matrix_family(...)` before running grid search or Optuna. This prevents preprocessings selected for `object_matrix` from being applied to `pixel_matrix`, and vice versa, unless the preprocessing appears in both PCA shortlist families.

### Notebook 04B — benchmark Optuna 8-tracks

04B est un benchmark par identifiant sur les sorties exhaustives de 04A. Les
objectifs actifs et leurs directions sont dérivés, pour chaque
`evaluation_track`, de `SIMCA_EVALUATION_TRACK_SPECS` dans
`SIMCA_OPTUNA_OBJECTIVE_SPECS`. Le nom historique
`SIMCA_OPTUNA_DIRECTIONS` reste inchangé uniquement pour préserver le hash du
protocole amont gelé ; 04B ne l'utilise pas pour construire ses études.

Les réglages centralisés sont notamment :

- `SIMCA_OPTUNA_N_TRIALS_PER_TRACK` et `SIMCA_OPTUNA_N_STARTUP_TRIALS` ;
- `SIMCA_OPTUNA_RANDOM_STATE`, avec un décalage stable E1–E8 ;
- `SIMCA_OPTUNA_SAMPLER_NAME` et `SIMCA_OPTUNA_SAMPLER_MULTIVARIATE` ;
- `SIMCA_OPTUNA_MIN_PARETO_RECALL` ;
- `SIMCA_OPTUNA_UNIFORM_RECALL_DELTA_TOLERANCE` ;
- `SIMCA_OPTUNA_TECHNICAL_PRUNE_STATUSES` ;
- `SIMCA_ABLATION_*` pour le plan apparié gelé avant la prochaine exécution
  8-tracks du batch 3.

`SIMCA_OPTUNA_REUSE_GRID_METRICS` doit rester vrai : le notebook ne possède
plus de chemin actif autorisant un refit ou un chargement du H5.

## Frozen Protocol Artifacts And Inference Plan

`scripts/freeze_protocol.py` calls the blocking validations in
`src.protocol_governance` and writes:

- `docs/protocol/protocol_manifest.parquet`: every curated scientific setting,
  its individual hash and the common configuration hash;
- `docs/protocol/protocol_checks.parquet`: blocking checks for batches, axes,
  tracks, thresholds, Pareto, hypotheses and contrast completeness;
- `docs/protocol/inference_plan.json`: the frozen H1-H4 analysis plan;
- `docs/protocol/planned_contrasts.parquet`: one row per planned estimand;
- `docs/protocol/protocol_lock.json`: semantic hashes and artifact checksums.

The plan uses `source_image` as its primary bootstrap unit. Pixels and objects
from the same image remain grouped. The confidence level is 95%, the default
bootstrap budget is 2,000 resamples, and multiplicity is controlled with Holm
within each H1-H4 family.

The prespecified practical equivalence zones are:

- 0.05 for absolute rate differences;
- 0.20 for standardized train-projection shift.

The hypotheses are:

- H1: object performance by training family at fixed object projection;
- H2: pixel and small-fragment performance by training family at fixed pixel
  projection;
- H3: training-family × projection interaction on standardized domain shift;
- H4: 3-way versus 2-way through risk-coverage analysis.

Existing outputs generated before this freeze are declared
`legacy_exploratory`. No claim that batch 4 remained unseen is made by the
protocol. Primary contrasts remain fixed prospectively from this version.

## SIMCA Output Table Policy

SIMCA notebooks write compact result tables with schemas centralized in
`experiment_config.py`. Legacy 04A-to-05 tables can additionally use
`src.workflows.simca_tables`; the 8-track 04C contract has dedicated schemas.

The table policy is:

- resolve pandas merge suffixes such as `_x` and `_y`;
- normalize common aliases such as `non_target_class` to `non_target_label`;
- preserve and, when possible, re-infer `matrix_family` from `matrix_method`, `training_matrix_id`, or `selection_track`;
- keep model-defining parameters in their effective form, for example `m_effective` and `balanced_pixel_strategy_effective`;
- drop columns that are entirely non-applicable for a given output;
- preserve stable empty schemas for known empty support tables, such as error logs and optional duplicate-refit outputs.

Scientific metric tables have stricter schemas than detailed object or pixel tables. Detailed projection tables may keep additional diagnostic columns because they are consumed by border/core and downstream visual checks.

Use:

```python
from src.workflows.simca_tables import compact_simca_table_for_path

save_parquet(compact_simca_table_for_path(df, output_path), output_path)
```

for SIMCA result files. When adding a new SIMCA output file, register its file name or suffix in `TABLE_KIND_BY_FILE_NAME` or `TABLE_KIND_BY_FILE_SUFFIX` if the table should have a strict schema. Leave it unregistered only when all non-empty columns are intentionally diagnostic payload.

Notebook 04C uses the `SIMCA_CONCAT_REFIT_*` settings. Its candidate cap must
remain `None`: a row-order cap would introduce an unregistered selection. The
pool consists of the 04A protocol Pareto front for supported tracks and the
diagnostic Pareto front for `unsupported_domain_shift` tracks. Optuna is only
provenance.

The expensive refit resumes through `SIMCA_CONCAT_REFIT_CHECKPOINT_ENABLED`
and `SIMCA_CONCAT_REFIT_RESUME_FROM_CHECKPOINT`. Checkpoint shards are accepted
only when all four hashes match:

- validation plan (`SIMCA_CONCAT_REFIT_VALIDATION_PLAN_KEYS`);
- expanded seed-level candidate pool;
- preregistered 04B ablation plan;
- immutable 03C spatial lock.

Continuous object/pixel predictions are stored once per
`projection_config_id`; locked candidate decisions are derived afterwards.
Map encoding, confidence level, border width, component IoU convention and the
absence of a smallest-fragment threshold are centralized respectively in
`SIMCA_CONCAT_REFIT_MAP_ENCODING`,
`SIMCA_CONCAT_REFIT_CONFIDENCE_LEVEL`,
`SIMCA_CONCAT_REFIT_BORDER_WIDTH`,
`SIMCA_CONCAT_REFIT_COMPONENT_MIN_IOU` and
`SIMCA_CONCAT_REFIT_SMALLEST_FRAGMENT_RECALL_MIN`.

## SIMCA Robustness Review Policy

Notebook 05 uses the following project-level settings from `src/experiment_config.py`:

- `SIMCA_ROBUSTNESS_RANDOM_STATES`: seeds used by the optional random-state stability refit.
- `SIMCA_ROBUSTNESS_MAX_STABILITY_CANDIDATES_PER_TRACK`: maximum candidates per track sent to the optional seed-stability panel.
- `SIMCA_ROBUSTNESS_PREFER_BALANCED_PIXELS_FOR_STABILITY`: prioritizes `balanced_pixels` candidates when the stability panel is limited.
- `SIMCA_ROBUSTNESS_BORDER_WIDTHS`: border widths tested when detailed pixel tables are available.
- `SIMCA_ROBUSTNESS_MIN_CORE_PIXELS`: minimum core pixels required before a core-only object decision is trusted.
- `SIMCA_ROBUSTNESS_PARETO_EPSILON`: numerical tolerance used by Pareto dominance checks.
- `SIMCA_ROBUSTNESS_WARNING_THRESHOLDS`: warning thresholds used to create notebook-05 flags.
- `SIMCA_ROBUSTNESS_2WAY_SCORE_WEIGHTS` and
  `SIMCA_ROBUSTNESS_3WAY_SCORE_WEIGHTS` are legacy notebook-05 settings. They
  are not allowed in the rewritten 00-04B selection protocol and must be
  removed when notebook 05 is refactored.
- `SIMCA_ROBUSTNESS_ABLATION_FACTOR_COLUMNS`: hyperparameters summarized in the ablation diagnostics.
- `SIMCA_ROBUSTNESS_2WAY_PARETO_MINIMIZE_COLUMNS` and `SIMCA_ROBUSTNESS_2WAY_PARETO_MAXIMIZE_COLUMNS`: 2-way Pareto objectives.
- `SIMCA_ROBUSTNESS_3WAY_PARETO_MINIMIZE_COLUMNS` and `SIMCA_ROBUSTNESS_3WAY_PARETO_MAXIMIZE_COLUMNS`: 3-way Pareto objectives.

Notebook 05 is a validation robustness review. It must not consume pure-test outputs and must not perform final model selection. Its outputs should be interpreted as diagnostic evidence for the later pure-test and final multi-model selection stages.

## Matrix And Pixel Sampling Defaults

- `M_BALANCED_PIXELS = 10`: number of sampled pixels per object for balanced
  pixel matrices. This retains 100% of calibration objects without
  replacement; `m=20` already excludes two objects.
- `BALANCED_PIXEL_STRATEGIES`: allowed balanced-pixel sampling strategies.
- `REPLACE_BALANCED_PIXELS`: whether balanced pixel sampling uses replacement.
- `CV_N_SPLITS`: default number of grouped CV splits.
- `CV_GROUP_COL`: default grouping column for grouped validation.

Matrix construction is formalized by `src.matrices.matrix_registry.MatrixOutput`, whose contract is `X`, `y`, `metadata`, and `wavelengths`. Existing notebooks can keep using `build_matrix()` as `X, y, metadata`; new code can request wavelengths with `return_wavelengths=True` or use `build_matrix_output(...)`.

## Notebook 02 Result-Table Contracts

Notebook 02 validates required output schemas before saving key result tables.

- `MATRIX_SUMMARY_REQUIRED_COLUMNS`: required columns for `results/02_matrices_<RESULTS_TAG>/matrix_summary.parquet`.
- `PREPROCESSING_SUMMARY_REQUIRED_COLUMNS`: required columns for `results/02_matrices_<RESULTS_TAG>/preprocessing_validation.parquet`.

The contract is enforced with `src.workflows.matrix_preprocessing.validate_required_columns(...)`.

## Notebook 03 PCA Protocol

The active PCA configuration is entirely centralized in
`src/experiment_config.py`.

### Data Roles

- `PCA_CALIBRATION_BATCHES`: batches 1–2; PCA and preprocessing are fitted here.
- `PCA_FORBIDDEN_BATCHES`: batches 3–4; blocked from fits, projections and diagnostics.
- `PCA_SAMPLE_KIND`: pure reference objects only.

### Representations And Stability

- `PCA_MATRIX_METHODS`, `PCA_BALANCED_M_VALUES` and
  `PCA_BALANCED_STRATEGIES`: the accepted task-15 representation universe.
- Preprocessing candidates are read from the accepted notebook-02 table; they
  are not reconstructed from a second configuration list.
- `PCA_N_COMPONENTS` and `PCA_DIAGNOSTIC_N_COMPONENTS`: retained and diagnostic dimensions.
- `PCA_STABILITY_SEEDS`, `PCA_STABILITY_N_SPLITS`,
  `PCA_STABILITY_N_BOOTSTRAP`, `PCA_STABILITY_GROUP_COL` and
  `PCA_STABILITY_BOOTSTRAP_GROUP_COL`: common grouped folds, sampling seeds,
  source-image bootstrap and subspace stability protocol.

### Relative And Pareto Selection

Notebook 03 does not compute a weighted selection score.

- `PCA_SELECTION_SEPARATION_QUANTILE = 0.50`
- `PCA_SELECTION_BATCH_STRICT_QUANTILE = 0.50`
- `PCA_SELECTION_BATCH_RELAXED_QUANTILE = 0.75`
- `PCA_SELECTION_PROJECTION_QUANTILE = 0.75`
- `PCA_SELECTION_INSTABILITY_QUANTILE = 0.75`
- `MAX_PCA_PREPROCESSINGS_PER_FAMILY = None` keeps the complete
  preprocessing-level Pareto front in notebook 03. Projection diagnostics,
  crowding and diversity do not participate in this selection.

`PCA_SELECTION_PROFILES` defines non-redundant maximize/minimize objectives for
the object and pixel families. The batch constraint uses the median by default
and can relax to the third quartile only when the strict pool is insufficient.
Projection constraints use grouped validation inside batches 1–2. No batch-3
metric is produced.

Visual decisions are stored only in the fingerprinted
`pca_artifact_review.parquet`; they are not hardcoded in configuration.
The notebook binds them to the current `run_fingerprint` while requiring the
exact SHA-256 of the human-reviewed PDF. A protocol-lock-only change can reuse
the decisions when the regenerated PDF is byte-identical; any PDF change is a
blocking request for a new review.

`build_pca_selection_flow_tables(...)` exposes the count entering, retained and
eliminated at the technical, relative-constraint and Pareto stages, plus the
first elimination stage and reason for every candidate. Notebook 03 validates
the capped shortlist before saving it.

### Minimal Output Contracts

- `PCA_SUMMARY_COLUMNS`: candidate-aware component-wise variance table.
- `PCA_SCORING_DIAGNOSTIC_COLUMNS`: candidate-aware long-form audit table.
- `PCA_SELECTED_PREPROCESSING_COLUMNS`: capped, hashed shortlist contract.
- `PCA_OUTPUT_FILENAMES`: four Parquet tables plus `pca_visual_review.pdf`.

When the PCA protocol or artifact review changes, rerun notebook 03 and all
downstream SIMCA notebooks that consume `pca_selected_preprocessings.parquet`.

## Notebook 03B Internal Calibration

Notebook 03B is configured entirely through the
`INTERNAL_CALIBRATION_*` constants in `src/experiment_config.py`.

The protocol lock is:

- batches 1–2 for grouped out-of-fold calibration;
- batch 3 forbidden because it is the later external validation batch;
- batch 4 forbidden because it is the pure-test batch;
- grouping by `source_image`;
- two folds preserving class, batch and relative object size. This is the
  largest complete image-level split supported by the four pure images.

The grid covers matrices, balanced-pixel `m=(10, 20)` and strategy, components
3–12, eight theoretical/empirical rule variants, alpha, SG settings, dilation
and multiple random seeds. `under_m_policy="exclude"` excludes small objects
from the affected training matrix but never from OOF projection. Empirical
rule limits are computed from target observations in the training part of the
current outer fold only.

For the active 03B run:

- `INTERNAL_CALIBRATION_RANDOM_SEEDS=(0, 1, 2)`;
- `INTERNAL_CALIBRATION_DILATION_RADII=(0,)`, because dilation is not
  identifiable on pure references;
- `INTERNAL_CALIBRATION_AVAILABLE_DILATION_RADII=(0, 2, 3, 5)` preserves the
  later candidate set;
- `fit_config_id`, `projection_config_id` and `evaluation_config_id` separate
  fitting, projection and decision identities;
- one fit feeds every authorised object/pixel projection without refitting;
- the direct 2-way threshold is the zero SIMCA margin;
- signed 3-way thresholds come from the centralized lower/upper quantiles and
  are evaluated by cross-fitting;
- the final manifest stores schema, protocol/PCA/contract hashes, row counts,
  columns and SHA-256 for every output table.

Risk constraints are grouped in `INTERNAL_CALIBRATION_RISK_PROFILES`. The
active exploratory profile is permissive; switching to `final_strict`
activates the stricter profile without changing notebook code. Compatibility
constants remain explicit in:

- `INTERNAL_CALIBRATION_MAX_FN_RATE`
- `INTERNAL_CALIBRATION_MAX_FP_RATE`
- `INTERNAL_CALIBRATION_MIN_BALANCED_ACCURACY`
- `INTERNAL_CALIBRATION_MIN_DECISION_RATE`
- `INTERNAL_CALIBRATION_MAX_TARGET_MISS_RATE`
- `INTERNAL_CALIBRATION_MAX_FALSE_ACCEPT_RATE`
- `INTERNAL_CALIBRATION_MAX_UNCERTAIN_RATE`
- `INTERNAL_CALIBRATION_MIN_COVERAGE`

No unconstrained fallback or weighted score is allowed in notebook 03B. The 16
canonical output names are defined in
`INTERNAL_CALIBRATION_OUTPUT_FILENAMES`. Object and pixel OOF projections are
produced in one pass; hyperparameters stay in configuration/domain tables and
are referenced from OOF rows by stable identifiers.
