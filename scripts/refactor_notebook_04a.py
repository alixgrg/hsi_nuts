"""Regenerate notebook 04A as the thin orchestrator for protocol tasks 27-28."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "04A_simca_grid_search.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


cells = [
    markdown(
        """
# 04A — Évaluation exhaustive du domaine SIMCA calibré (tâches 27–28)

Ce notebook réutilise exclusivement le domaine, les folds, les seuils et les
prédictions OOF verrouillés par 03B, puis le statut de domaine et le verrou
spatial de 03C. Il ne réajuste aucun modèle et ne suggère aucun seuil.

Les cinq statuts restent indépendants : calculabilité technique,
acceptabilité, éligibilité de domaine, équivalence et appartenance au Pareto.
Les tracks non soutenus sont conservés comme résultats scientifiques.
"""
    ),
    markdown("## A — Initialisation et chemins centralisés"),
    code(
        r'''
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

CURRENT_DIR = Path.cwd().resolve()
if (CURRENT_DIR / "src").exists():
    PROJECT_ROOT = CURRENT_DIR
elif (CURRENT_DIR.parent / "src").exists():
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    raise RuntimeError("Launch the notebook from the project root or notebooks/.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pd.set_option("display.max_columns", 24)
pd.set_option("display.max_rows", 30)

from src import experiment_config as expcfg
from src.io.database_h5 import load_nir_uco_h5
from src.protocol_governance import sha256_file
from src.utils import load_parquet, save_parquet
from src.workflows.protocol_audit import assert_no_forbidden_score_columns
from src.workflows.simca_grid_evaluation import (
    run_exhaustive_locked_grid_evaluation,
)
from src.workflows.spatial_postprocessing_calibration import (
    verify_spatial_postprocessing_lock,
)

%load_ext autoreload
%autoreload 2

print("Python:", platform.python_version())
print("PROJECT_ROOT:", PROJECT_ROOT)
'''
    ),
    code(
        r'''
DB_H5_PATH = PROJECT_ROOT.joinpath(*expcfg.DATABASE_H5_RELATIVE_PATH)
RESULTS_TAG = (
    f"{int(expcfg.WAVELENGTH_WINDOW_MIN_NM)}_{int(expcfg.WAVELENGTH_WINDOW_MAX_NM)}"
    if expcfg.USE_WAVELENGTH_WINDOW
    else expcfg.DEFAULT_RESULTS_TAG
)
CALIBRATION_RESULTS_DIR = (
    PROJECT_ROOT / "results"
    / f"{expcfg.INTERNAL_CALIBRATION_RESULTS_DIR_PREFIX}_{RESULTS_TAG}"
)
DOMAIN_CALIBRATION_RESULTS_DIR = (
    PROJECT_ROOT / "results"
    / f"{expcfg.DOMAIN_SPATIAL_CALIBRATION_RESULTS_DIR_PREFIX}_{RESULTS_TAG}"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "results"
    / f"{expcfg.SIMCA_GRID_SEARCH_RESULTS_DIR_PREFIX}_{RESULTS_TAG}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_03B = {
    key: CALIBRATION_RESULTS_DIR / expcfg.INTERNAL_CALIBRATION_OUTPUT_FILENAMES[key]
    for key in (
        "calibration_domain", "folds", "track_contracts",
        "oof_object_predictions", "oof_pixel_predictions", "checkpoint_manifest",
    )
}
INPUT_03C = {
    key: DOMAIN_CALIBRATION_RESULTS_DIR
    / expcfg.DOMAIN_SPATIAL_CALIBRATION_OUTPUT_FILENAMES[key]
    for key in (
        "projection_eligibility", "spatial_calibration_metrics",
        "fragment_size_classes", "spatial_postprocessing_lock",
    )
}
OUTPUT_PATHS = {
    key: OUTPUT_DIR / filename
    for key, filename in expcfg.SIMCA_GRID_SEARCH_OUTPUT_FILENAMES.items()
}

print("03B:", CALIBRATION_RESULTS_DIR)
print("03C:", DOMAIN_CALIBRATION_RESULTS_DIR)
print("04A:", OUTPUT_DIR)
'''
    ),
    markdown("## B — Contrats 03B/03C et contrôles bloquants"),
    code(
        r'''
required_inputs = [DB_H5_PATH, *INPUT_03B.values(), *INPUT_03C.values()]
missing_inputs = [str(path) for path in required_inputs if not path.exists()]
if missing_inputs:
    raise FileNotFoundError(f"Missing upstream inputs: {missing_inputs}")

calibration_domain_df = load_parquet(INPUT_03B["calibration_domain"])
calibration_folds_df = load_parquet(INPUT_03B["folds"])
track_contracts_df = load_parquet(INPUT_03B["track_contracts"])
oof_object_predictions_df = load_parquet(INPUT_03B["oof_object_predictions"])
oof_pixel_predictions_df = load_parquet(INPUT_03B["oof_pixel_predictions"])
projection_eligibility_df = load_parquet(INPUT_03C["projection_eligibility"])
spatial_metrics_df = load_parquet(INPUT_03C["spatial_calibration_metrics"])
fragment_size_classes_df = load_parquet(INPUT_03C["fragment_size_classes"])
calibration_manifest = json.loads(
    INPUT_03B["checkpoint_manifest"].read_text(encoding="utf-8")
)
spatial_lock = json.loads(
    INPUT_03C["spatial_postprocessing_lock"].read_text(encoding="utf-8")
)

verify_spatial_postprocessing_lock(
    spatial_lock,
    spatial_metrics_df,
    fragment_size_classes_df,
)

def top_level_shard(name):
    candidates = [
        shard for shard in calibration_manifest["shards"]
        if shard["name"] == name
        and Path(shard["path"]).parent.resolve() == CALIBRATION_RESULTS_DIR.resolve()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one top-level 03B shard for {name!r}.")
    return candidates[0]

for name in (
    "calibration_domain", "folds", "track_contracts",
    "oof_object_predictions", "oof_pixel_predictions",
):
    shard = top_level_shard(name)
    if sha256_file(INPUT_03B[name]) != shard["file_sha256"]:
        raise RuntimeError(f"03B hash mismatch for {name}.")

domain_hashes = set(calibration_domain_df["protocol_hash"].astype(str))
if domain_hashes != {str(calibration_manifest["protocol_hash"])}:
    raise RuntimeError("03B domain and manifest protocol hashes differ.")
if set(projection_eligibility_df["protocol_hash"].astype(str)) != domain_hashes:
    raise RuntimeError("03C eligibility does not match the 03B protocol hash.")
if str(spatial_lock["protocol_hash"]) not in domain_hashes:
    raise RuntimeError("03C spatial lock does not match the 03B protocol hash.")
if projection_eligibility_df["evaluation_track"].astype(str).nunique() != 8:
    raise RuntimeError("03C must diagnose exactly the eight evaluation tracks.")

fold_batches = set(
    pd.to_numeric(calibration_folds_df["batch"], errors="raise").astype(int)
)
if not fold_batches <= set(expcfg.INTERNAL_CALIBRATION_BATCHES):
    raise RuntimeError(f"Unexpected calibration batch: {sorted(fold_batches)}")
if fold_batches.intersection(expcfg.INTERNAL_CALIBRATION_FORBIDDEN_BATCHES):
    raise RuntimeError("A forbidden batch entered notebook 04A.")

unsupported_tracks = sorted(
    projection_eligibility_df.loc[
        ~projection_eligibility_df["eligibility_status"].isin(
            expcfg.SIMCA_GRID_SUPPORTED_ELIGIBILITY_STATUSES
        ),
        "evaluation_track",
    ].astype(str)
)
spatial_supported_tracks = sorted(
    set(
        spatial_metrics_df.loc[
            spatial_metrics_df["is_locked_candidate"].astype(bool),
            "evaluation_track",
        ].astype(str)
    )
)

print("Domain rows:", len(calibration_domain_df))
print("Unique calibrated configurations:", calibration_domain_df["calibration_id"].nunique())
print("Unsupported tracks retained:", unsupported_tracks)
print("Tracks using the locked spatial map:", spatial_supported_tracks)
display(
    calibration_domain_df.groupby(
        ["evaluation_track", "track_id", "decision_mode", "projection_level"],
        as_index=False,
    ).agg(
        n_domain_rows=("domain_config_id", "nunique"),
        n_calibrations=("calibration_id", "nunique"),
        n_seeds=("random_state", "nunique"),
    )
)
'''
    ),
    markdown(
        """
## C — Évaluation exhaustive verrouillée

Les opérations sur les observations sont vectorisées. La boucle restante est
au niveau des configurations et, pour la morphologie, des images : elle est
nécessaire car chaque configuration possède ses propres seuils et chaque carte
sa propre géométrie. Les erreurs sont capturées par configuration dans
`technical_audit.parquet`.
"""
    ),
    code(
        r'''
if expcfg.SIMCA_GRID_SEARCH_RUN:
    _, image_db = load_nir_uco_h5(
        DB_H5_PATH,
        reconstruct_heavy_object_arrays=False,
        batches=expcfg.INTERNAL_CALIBRATION_BATCHES,
    )
    grid_outputs = run_exhaustive_locked_grid_evaluation(
        calibration_domain_df,
        oof_object_predictions_df,
        oof_pixel_predictions_df,
        projection_eligibility_df,
        image_db=image_db,
        spatial_lock=spatial_lock,
        spatial_supported_tracks=(
            spatial_supported_tracks if expcfg.SIMCA_GRID_APPLY_SPATIAL_LOCK else ()
        ),
    )
else:
    grid_outputs = {
        key: load_parquet(path)
        for key, path in OUTPUT_PATHS.items()
        if key != "protocol"
    }

grid_configurations_df = grid_outputs["configurations"]
grid_fold_metrics_df = grid_outputs["fold_metrics"]
grid_threshold_metrics_df = grid_outputs["threshold_metrics"]
grid_pareto_reference_df = grid_outputs["pareto_reference"]
technical_audit_df = grid_outputs["technical_audit"]
duplicate_groups_df = grid_outputs["duplicate_groups"]
calculable_not_acceptable_df = grid_outputs["calculable_not_acceptable"]

print("Fold/image metrics:", grid_fold_metrics_df.shape)
print("Aggregated metrics:", grid_threshold_metrics_df.shape)
print("Retained representatives:", len(grid_configurations_df))
display(
    technical_audit_df.groupby(
        ["evaluation_track", "technical_status", "acceptability_status",
         "eligibility_status"],
        dropna=False,
    ).size().rename("n_domain_rows").reset_index()
)
'''
    ),
    markdown("## D — Exhaustivité, Pareto par track et déduplication"),
    code(
        r'''
domain_ids = set(calibration_domain_df["domain_config_id"].astype(str))
audit_ids = set(technical_audit_df["domain_config_id"].astype(str))
if domain_ids != audit_ids or technical_audit_df["domain_config_id"].duplicated().any():
    raise RuntimeError("Every 03B domain row must have exactly one 04A audit row.")

track_summary = grid_pareto_reference_df.loc[
    grid_pareto_reference_df["row_type"].eq("track_summary")
]
if set(track_summary["evaluation_track"].astype(str)) != set(
    expcfg.SIMCA_EVALUATION_TRACKS
):
    raise RuntimeError("The Pareto reference must retain all eight tracks.")

configuration_pareto = grid_pareto_reference_df.loc[
    grid_pareto_reference_df["row_type"].eq("configuration")
]
display(
    configuration_pareto.groupby("evaluation_track", as_index=False).agg(
        n_calculable=("calibration_id", "nunique"),
        n_diagnostic_pareto=("diagnostic_pareto_front", "sum"),
        n_protocol_pareto=("protocol_pareto_front", "sum"),
    )
)
display(
    track_summary[
        ["track_id", "evaluation_track", "eligibility_status",
         "pareto_exclusion_reason"]
    ]
)

if duplicate_groups_df.empty:
    print("No exact configuration/output duplicate detected.")
else:
    display(duplicate_groups_df)
if not calculable_not_acceptable_df.empty:
    display(
        calculable_not_acceptable_df[
            ["track_id", "calibration_id", "failure_reason"]
        ].head(30)
    )
'''
    ),
    markdown("## E — Persistance compacte et provenance"),
    code(
        r'''
score_column_audit_df = assert_no_forbidden_score_columns(grid_outputs)
display(score_column_audit_df)

if expcfg.SIMCA_GRID_SEARCH_RUN:
    for key, table in grid_outputs.items():
        save_parquet(table, OUTPUT_PATHS[key])

output_hashes = {
    key: sha256_file(path)
    for key, path in OUTPUT_PATHS.items()
    if key != "protocol" and path.exists()
}
protocol = {
    "notebook": "04A_simca_grid_search",
    "protocol_task_ids": [27, 28],
    "protocol_version": str(calibration_manifest["protocol_version"]),
    "protocol_hash": str(calibration_manifest["protocol_hash"]),
    "results_tag": RESULTS_TAG,
    "calibration_batches": list(expcfg.INTERNAL_CALIBRATION_BATCHES),
    "forbidden_batches": list(expcfg.INTERNAL_CALIBRATION_FORBIDDEN_BATCHES),
    "threshold_policy": "locked_from_03B_no_refit_no_resuggestion",
    "spatial_policy": "locked_from_03C_preserve_uncertainty",
    "selection_policy": "hard_constraints_then_pareto_by_evaluation_track",
    "weighted_score_used": False,
    "pareto_epsilon": float(expcfg.SIMCA_GRID_PARETO_EPSILON),
    "supported_eligibility_statuses": list(
        expcfg.SIMCA_GRID_SUPPORTED_ELIGIBILITY_STATUSES
    ),
    "unsupported_tracks_retained": unsupported_tracks,
    "spatial_supported_tracks": spatial_supported_tracks,
    "track_objectives": {
        track: {
            "minimize": list(spec["pareto_minimize"]),
            "maximize": list(spec["pareto_maximize"]),
        }
        for track, spec in expcfg.SIMCA_EVALUATION_TRACK_SPECS.items()
    },
    "risk_constraints": expcfg.SIMCA_SEARCH_CONSTRAINTS,
    "input_sha256": {
        **{f"03B_{key}": sha256_file(path) for key, path in INPUT_03B.items()},
        **{f"03C_{key}": sha256_file(path) for key, path in INPUT_03C.items()},
    },
    "output_sha256": output_hashes,
    "n_domain_rows": int(len(calibration_domain_df)),
    "n_calibrations": int(calibration_domain_df["calibration_id"].nunique()),
    "n_technical_errors": int(technical_audit_df["calculable"].eq(False).sum()),
    "n_calculable_not_acceptable": int(len(calculable_not_acceptable_df)),
    "n_duplicate_groups": int(len(duplicate_groups_df)),
    "n_retained_representatives": int(len(grid_configurations_df)),
}
OUTPUT_PATHS["protocol"].write_text(
    json.dumps(protocol, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Saved:")
for path in OUTPUT_PATHS.values():
    print(" -", path)
'''
    ),
    markdown(
        """
## Lecture des sorties

- `grid_fold_metrics.parquet` : métriques par seed, fold et image, avec la
  variante de carte explicite.
- `grid_threshold_metrics.parquet` : agrégation par `calibration_id`, comptes,
  dispersion, acceptabilité et éligibilité.
- `grid_pareto_reference.parquet` : Pareto diagnostic et Pareto protocolaire
  séparés dans chacun des huit tracks, y compris les tracks non soutenus.
- `technical_audit.parquet` : exactement une ligne par `domain_config_id`.
- `duplicate_groups.parquet` : équivalences exactes et représentant déterminé
  lexicalement, jamais par performance.
- `calculable_not_acceptable.parquet` : configurations techniquement valides
  qui échouent aux contraintes, conservées pour l’audit.
- `grid_configurations.parquet` : domaine principal dédupliqué avec toutes les
  seeds et identifiants d’origine sérialisés.
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "protocol_scope": "tasks_27_28",
    },
)
nbf.write(notebook, TARGET)
print(TARGET)
