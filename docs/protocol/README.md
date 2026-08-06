# Frozen scientific protocol

This directory contains the version-controlled artifacts for protocol
`8tracks_v3` (amendement daté du 3 août 2026 pour les diagnostics de domaine
et le verrou spatial des tâches 25–26).

The bundle is generated from `src/experiment_config.py` by:

```powershell
conda run -n hsi-nuts python scripts\freeze_protocol.py --overwrite
```

Verify the frozen bundle without rewriting it:

```powershell
conda run -n hsi-nuts python scripts\freeze_protocol.py --verify
```

Files:

- `protocol_manifest.parquet`: task-01 scientific configuration manifest;
- `protocol_checks.parquet`: blocking contract checks;
- `inference_plan.json`: task-02 H1-H4 inference plan;
- `planned_contrasts.parquet`: prespecified contrast table;
- `protocol_lock.json`: semantic hashes and file checksums.
- `qc_review_decisions.parquet`: versioned human QC decisions, deliberately
  stored outside scientific configuration.

Do not edit generated files manually. Any scientific amendment requires a new
`PROTOCOL_VERSION`, a dated justification, regeneration of the full bundle and
rerunning all affected notebooks.

Outputs produced before this freeze are supporting or exploratory evidence.
They cannot redefine the primary contrasts stored here.

## Addendum 04B avant la prochaine exécution 8-tracks du batch 3

Le benchmark Optuna des tâches 29–30 possède un `search_plan_hash` enfant du
`protocol_hash` ci-dessus. Il ne réécrit pas ce bundle parent et ne modifie
aucun résultat 03B/03C/04A déjà calculé. Son manifeste
`optuna_protocol.json` relie les empreintes 04A/03C, les huit études et le
fichier `preregistered_ablation_plan.parquet`.

Le plan d'ablation est prospectif pour la prochaine exécution 8-tracks. Des
sorties legacy de 04C/05 ayant déjà traité le batch 3, aucune revendication de
première ouverture historique ou de cécité absolue n'est faite. 04C bloque
avant chargement du batch 3 si le plan ou son hash est absent.
