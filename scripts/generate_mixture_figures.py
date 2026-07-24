"""Generate report figures and compact tables for notebook 07 mixture outputs.

The script consumes the parquet files written by
``notebooks/07_simca_mixture_application.ipynb``. It does not refit models,
does not recompute thresholds, and does not change notebook results.

Outputs are written under:
``results/07_simca_mixture_application_<RESULTS_TAG>/figures``

Examples
--------
python scripts/generate_mixture_figures.py --results-tag non_noisy_all

python scripts/generate_mixture_figures.py --results-tag non_noisy_all --formats html png

python scripts/generate_mixture_figures.py --results-tag non_noisy_all --skip-spatial
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


TRACK_ORDER = (
    "object_matrix_2way",
    "object_matrix_3way",
    "pixel_matrix_2way",
    "pixel_matrix_3way",
)

CONFIG_COLUMNS = (
    "selected_config_id",
    "candidate_id",
    "assigned_selection_track",
    "selection_track",
    "matrix_family",
    "decision_mode",
    "metric_level",
    "matrix_method",
    "preprocessing",
    "rule_for_refit",
    "n_components",
    "alpha",
    "object_threshold",
    "balanced_pixel_strategy_effective",
    "three_way_lower_threshold",
    "three_way_upper_threshold",
    "final_rank_in_track",
    "pareto_tier",
    "pareto_rank_in_track",
    "is_pareto_front",
    "selection_reason",
    "previous_flags",
)

TWO_WAY_METRICS = ("fn_rate", "fp_rate", "balanced_accuracy")
THREE_WAY_METRICS = (
    "target_miss_rate",
    "non_target_false_accept_rate",
    "uncertain_rate",
    "coverage_rate",
    "decided_balanced_accuracy",
)


@dataclass(frozen=True)
class MixturePaths:
    """Canonical notebook-07 result paths used by this script."""

    root: Path
    selected_configs: Path
    metrics_long: Path
    metrics_2way_object: Path
    metrics_2way_pixel: Path
    metrics_3way_object: Path
    object_image_diagnostics: Path
    pixel_image_diagnostics: Path
    object_3way_image_diagnostics: Path
    pixel_errors_by_image: Path
    objects: Path
    pixels: Path
    objects_3way: Path
    summary: Path
    guardrails: Path
    protocol: Path
    errors: Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from an arbitrary starting path."""
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "src").exists() and (candidate / "notebooks").exists():
            return candidate
    raise RuntimeError("Could not find the project root. Pass --project-root.")


def build_mixture_paths(results_dir: Path) -> MixturePaths:
    return MixturePaths(
        root=results_dir,
        selected_configs=results_dir / "mixture_selected_configs.parquet",
        metrics_long=results_dir / "mixture_metrics_long.parquet",
        metrics_2way_object=results_dir / "mixture_2way_object_metrics.parquet",
        metrics_2way_pixel=results_dir / "mixture_2way_pixel_metrics.parquet",
        metrics_3way_object=results_dir / "mixture_3way_object_metrics.parquet",
        object_image_diagnostics=results_dir / "mixture_object_diagnostics_by_image.parquet",
        pixel_image_diagnostics=results_dir / "mixture_pixel_diagnostics_by_image.parquet",
        object_3way_image_diagnostics=results_dir / "mixture_3way_object_diagnostics_by_image.parquet",
        pixel_errors_by_image=results_dir / "mixture_pixel_errors_by_image.parquet",
        objects=results_dir / "mixture_objects.parquet",
        pixels=results_dir / "mixture_pixels.parquet",
        objects_3way=results_dir / "mixture_3way_objects.parquet",
        summary=results_dir / "mixture_summary.parquet",
        guardrails=results_dir / "mixture_guardrails.parquet",
        protocol=results_dir / "mixture_protocol.parquet",
        errors=results_dir / "mixture_errors.parquet",
    )


def require_existing(paths: Iterable[Path]) -> None:
    missing = [Path(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required notebook-07 output file(s):\n"
            + "\n".join(str(path) for path in missing)
        )


def read_table(path: Path, *, required: bool = False) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    if required:
        raise FileNotFoundError(f"Required table not found: {path}")
    print(f"[INFO] Optional table not found: {path}")
    return pd.DataFrame()


def compact_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[col for col in columns if col])
    return df[[col for col in columns if col in df.columns]].copy()


def write_table_bundle(df: pd.DataFrame, stem: Path) -> dict[str, Path]:
    """Write a table as parquet and CSV for quick inspection."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    if stem.suffix:
        stem = stem.with_suffix("")
    saved: dict[str, Path] = {}
    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    saved["parquet"] = parquet_path
    saved["csv"] = csv_path
    return saved


def track_col(df: pd.DataFrame) -> str:
    if "assigned_selection_track" in df.columns:
        return "assigned_selection_track"
    if "selection_track" in df.columns:
        return "selection_track"
    raise KeyError("Expected assigned_selection_track or selection_track.")


def add_track_order(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    col = track_col(out)
    order_map = {track: index for index, track in enumerate(TRACK_ORDER)}
    out["_track_order"] = out[col].astype(str).map(order_map).fillna(len(order_map)).astype(int)
    return out


def sort_configs(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = add_track_order(df)
    sort = [col for col in ("_track_order", "final_rank_in_track", "pareto_tier", "pareto_rank_in_track") if col in out.columns]
    if sort:
        out = out.sort_values(sort, ascending=True, na_position="last")
    return out.drop(columns=["_track_order"], errors="ignore").reset_index(drop=True)


def select_report_configs(
    selected_configs: pd.DataFrame,
    *,
    n_configs_per_track: int,
    max_configs: int | None,
) -> pd.DataFrame:
    """Pick a small, deterministic panel for per-configuration spatial figures."""
    if selected_configs.empty:
        return pd.DataFrame()
    d = sort_configs(selected_configs)
    col = track_col(d)
    selected = (
        d.groupby(col, group_keys=False, dropna=False)
        .head(int(n_configs_per_track))
        .reset_index(drop=True)
    )
    if max_configs is not None:
        selected = selected.head(int(max_configs)).copy()
    return selected.reset_index(drop=True)


def normalize_label(
    value,
    *,
    target_class: str = "peanut",
    non_target_label: str = "almond",
    uncertain_label: str = "uncertain",
) -> str:
    """Stable display labels for binary and three-way decisions."""
    if pd.isna(value):
        return "missing"
    raw = str(value).strip()
    val = raw.lower()
    if val in {str(target_class).lower(), "target", "true", "1", "peanut", "peanut_only"}:
        return str(target_class)
    if val in {
        str(non_target_label).lower(),
        "non_target",
        "non-target",
        "non_peanut",
        "false",
        "0",
        "almond",
        "almond_only",
    }:
        return str(non_target_label)
    if val in {str(uncertain_label).lower(), "uncertain", "ambiguous", "indeterminate"}:
        return str(uncertain_label)
    return raw


def add_plot_labels(
    df: pd.DataFrame,
    *,
    target_class: str,
    non_target_label: str,
    uncertain_label: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    label_kwargs = {
        "target_class": target_class,
        "non_target_label": non_target_label,
        "uncertain_label": uncertain_label,
    }
    for col in ("true_label_object", "predicted_label_object", "predicted_label_pixel", "decision_3way"):
        if col in out.columns:
            out[f"{col}_plot"] = out[col].map(lambda value: normalize_label(value, **label_kwargs))
    return out


def config_folder_name(row: Mapping[str, object]) -> str:
    from src.visualization.common import sanitize_filename

    config_id = sanitize_filename(row.get("selected_config_id", "configuration"))
    track = sanitize_filename(row.get("assigned_selection_track", row.get("selection_track", "track")))
    rank = row.get("final_rank_in_track", None)
    if rank is not None and pd.notna(rank):
        try:
            rank_part = f"rank{int(rank):02d}"
        except Exception:
            rank_part = f"rank-{sanitize_filename(rank)}"
    else:
        rank_part = "rankNA"
    return "__".join([rank_part, config_id, track])[:110]


def save_fig(
    fig: go.Figure | None,
    stem: Path,
    *,
    formats: Sequence[str],
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
    strict: bool = False,
) -> dict[str, Path]:
    if fig is None:
        print(f"[INFO] Skipped empty figure: {stem}")
        return {}
    from src.visualization.common import save_figure_bundle

    saved = save_figure_bundle(
        fig,
        stem,
        formats=formats,
        width=width,
        height=height,
        scale=scale,
        strict=strict,
    )
    if saved:
        print("  saved", ", ".join(str(path) for path in saved.values()))
    else:
        print(f"  [WARNING] No figure file written for {stem}")
    return saved


def build_parameter_tendency_table(
    configs_df: pd.DataFrame,
    *,
    parameter_cols: Sequence[str] = (
        "matrix_method",
        "preprocessing",
        "rule_for_refit",
        "n_components",
        "object_threshold",
        "balanced_pixel_strategy_effective",
        "three_way_lower_threshold",
        "three_way_upper_threshold",
    ),
) -> pd.DataFrame:
    if configs_df.empty:
        return pd.DataFrame()
    col = track_col(configs_df)
    rows: list[dict[str, object]] = []
    for track, group in configs_df.groupby(col, dropna=False):
        denom = max(len(group), 1)
        for parameter in parameter_cols:
            if parameter not in group.columns:
                continue
            values = group[parameter].astype("object").where(group[parameter].notna(), "not_applicable")
            counts = values.astype(str).value_counts(dropna=False)
            for value, count in counts.items():
                rows.append(
                    {
                        "assigned_selection_track": str(track),
                        "matrix_family": str(group["matrix_family"].dropna().iloc[0])
                        if "matrix_family" in group.columns and group["matrix_family"].notna().any()
                        else "unknown",
                        "parameter": parameter,
                        "value": str(value),
                        "n_models": int(count),
                        "track_size": int(denom),
                        "top_rate": float(count) / float(denom),
                    }
                )
    return pd.DataFrame(rows)


def build_binary_confusion_table(
    df: pd.DataFrame,
    *,
    true_col: str,
    pred_col: str,
    group_cols: Sequence[str] = (),
    target_class: str = "peanut",
    non_target_label: str = "almond",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    missing = [col for col in (true_col, pred_col) if col not in df.columns]
    if missing:
        return pd.DataFrame()
    group_cols = [col for col in group_cols if col in df.columns]
    d = df[df[true_col].notna() & df[pred_col].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["true_label_2way"] = d[true_col].map(
        lambda value: normalize_label(value, target_class=target_class, non_target_label=non_target_label)
    )
    d["predicted_label_2way"] = d[pred_col].map(
        lambda value: normalize_label(value, target_class=target_class, non_target_label=non_target_label)
    )
    labels = [non_target_label, target_class]
    rows: list[dict[str, object]] = []
    grouped = d.groupby(group_cols, dropna=False) if group_cols else [((), d)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        n_group = len(group)
        for true_label in labels:
            true_group = group[group["true_label_2way"].astype(str).eq(str(true_label))]
            n_true = len(true_group)
            for pred_label in labels:
                n_cell = int(true_group["predicted_label_2way"].astype(str).eq(str(pred_label)).sum())
                row = dict(base)
                row.update(
                    true_label_2way=true_label,
                    predicted_label_2way=pred_label,
                    n=n_cell,
                    n_true_label=int(n_true),
                    n_group=int(n_group),
                    row_rate=n_cell / n_true if n_true else np.nan,
                    global_rate=n_cell / n_group if n_group else np.nan,
                )
                rows.append(row)
    return pd.DataFrame(rows)


def add_object_count_columns(object_diag_df: pd.DataFrame) -> pd.DataFrame:
    """Add true/predicted target object counts from TP/FN/FP counts."""
    if object_diag_df is None or object_diag_df.empty:
        return pd.DataFrame() if object_diag_df is None else object_diag_df.copy()
    out = object_diag_df.copy()
    if {"tp", "fn"}.issubset(out.columns):
        out["n_true_target_objects"] = pd.to_numeric(out["tp"], errors="coerce") + pd.to_numeric(
            out["fn"], errors="coerce"
        )
    if {"tp", "fp"}.issubset(out.columns):
        out["n_predicted_target_objects"] = pd.to_numeric(out["tp"], errors="coerce") + pd.to_numeric(
            out["fp"], errors="coerce"
        )
    return out


def build_pixel_3way_view(pixel_df: pd.DataFrame, object_3way_df: pd.DataFrame) -> pd.DataFrame:
    """Attach object-level 3-way decisions to pixels for overlay diagnostics."""
    required_pixel = {"selected_config_id", "source_image", "object_id"}
    required_object = {"selected_config_id", "source_image", "object_id", "decision_3way"}
    if pixel_df.empty or object_3way_df.empty:
        return pd.DataFrame()
    if not required_pixel.issubset(pixel_df.columns) or not required_object.issubset(object_3way_df.columns):
        return pd.DataFrame()
    decision_cols = [
        col
        for col in (
            "selected_config_id",
            "source_image",
            "object_id",
            "decision_3way",
            "three_way_confidence",
            "three_way_margin",
        )
        if col in object_3way_df.columns
    ]
    decisions = object_3way_df[decision_cols].drop_duplicates(
        ["selected_config_id", "source_image", "object_id"]
    )
    return pixel_df.merge(
        decisions,
        on=["selected_config_id", "source_image", "object_id"],
        how="left",
        validate="many_to_one",
    )


def choose_images(
    image_metrics_df: pd.DataFrame,
    *,
    config_id: str,
    n_images: int,
    mode: str,
) -> pd.DataFrame:
    """Choose easy, hard, or mixed images from per-image metrics."""
    if image_metrics_df is None or image_metrics_df.empty or "source_image" not in image_metrics_df.columns:
        return pd.DataFrame()
    d = image_metrics_df.copy()
    if "selected_config_id" in d.columns:
        d = d[d["selected_config_id"].astype(str).eq(str(config_id))].copy()
    if d.empty:
        return pd.DataFrame()
    for col in set(TWO_WAY_METRICS + THREE_WAY_METRICS):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    hard_cols = [
        col
        for col in (
            "fn_rate",
            "fp_rate",
            "target_miss_rate",
            "non_target_false_accept_rate",
            "uncertain_rate",
            "balanced_accuracy",
            "coverage_rate",
        )
        if col in d.columns
    ]
    if not hard_cols:
        return d.drop_duplicates("source_image").head(int(n_images)).copy()

    hard_ascending = []
    for col in hard_cols:
        hard_ascending.append(col in {"balanced_accuracy", "coverage_rate"})
    hard = d.sort_values(hard_cols, ascending=hard_ascending, na_position="last")
    easy = d.sort_values(hard_cols, ascending=[not value for value in hard_ascending], na_position="last")

    if mode == "hardest":
        out = hard.head(int(n_images)).copy()
        out["image_selection_reason"] = "hardest"
    elif mode == "easiest":
        out = easy.head(int(n_images)).copy()
        out["image_selection_reason"] = "easiest"
    else:
        n_hard = int(np.ceil(int(n_images) / 2))
        n_easy = int(n_images) - n_hard
        h = hard.head(n_hard).copy()
        h["image_selection_reason"] = "hardest"
        e = easy.head(n_easy).copy()
        e["image_selection_reason"] = "easiest"
        out = pd.concat([h, e], ignore_index=True)
    return out.drop_duplicates("source_image").reset_index(drop=True)


def filter_config(df: pd.DataFrame, config_id: str) -> pd.DataFrame:
    if df is None or df.empty or "selected_config_id" not in df.columns:
        return pd.DataFrame()
    return df[df["selected_config_id"].astype(str).eq(str(config_id))].copy()


def sample_for_qt2(
    df: pd.DataFrame,
    *,
    label_col: str,
    max_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    max_rows = int(max_rows)
    if len(df) <= max_rows:
        return df.copy()
    if label_col in df.columns:
        n_groups = max(int(df[label_col].nunique(dropna=False)), 1)
        n_per_group = max(1, max_rows // n_groups)
        return (
            df.groupby(label_col, group_keys=False, dropna=False)
            .apply(lambda group: group.sample(n=min(len(group), n_per_group), random_state=random_state))
            .reset_index(drop=True)
        )
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def resolve_label_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def metric_stage_inputs(project_root: Path, results_tag: str) -> list[tuple[str, Path]]:
    """Optional upstream metric tables used for stage comparison figures."""
    return [
        (
            "validation_refit",
            project_root
            / "results"
            / f"04C_simca_concat_refit_{results_tag}"
            / "validation_refit_metrics_long.parquet",
        ),
        (
            "robustness_primary",
            project_root
            / "results"
            / f"05_simca_validation_robustness_{results_tag}"
            / "robustness_primary_metrics.parquet",
        ),
        (
            "pure_test",
            project_root
            / "results"
            / f"06A_simca_pure_test_{results_tag}"
            / "pure_test_metrics_long.parquet",
        ),
    ]


def build_stage_comparison_table(
    *,
    project_root: Path,
    results_tag: str,
    mixture_metrics_df: pd.DataFrame,
    selected_config_ids: Sequence[str],
) -> pd.DataFrame:
    selected = set(map(str, selected_config_ids))
    parts: list[pd.DataFrame] = []
    for fallback_stage, path in metric_stage_inputs(project_root, results_tag):
        if not path.exists():
            print(f"[INFO] Optional stage metric table not found: {path}")
            continue
        df = pd.read_parquet(path)
        if "selected_config_id" not in df.columns:
            continue
        df = df[df["selected_config_id"].astype(str).isin(selected)].copy()
        if df.empty:
            continue
        if "evaluation_stage" not in df.columns:
            df["evaluation_stage"] = fallback_stage
        parts.append(df)
    if not mixture_metrics_df.empty:
        parts.append(mixture_metrics_df[mixture_metrics_df["selected_config_id"].astype(str).isin(selected)].copy())
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    if "assigned_selection_track" not in out.columns and "selection_track" in out.columns:
        out["assigned_selection_track"] = out["selection_track"]
    return out


def make_global_figures(
    *,
    tables: Mapping[str, pd.DataFrame],
    output_root: Path,
    formats: Sequence[str],
    top_n_models: int,
    strict: bool,
    project_root: Path,
    results_tag: str,
    target_class: str,
    non_target_label: str,
    uncertain_label: str,
) -> list[dict[str, object]]:
    """Create global report figures and compact report tables."""
    from src.visualization.plot_decision import (
        plot_binary_confusion_heatmap,
        plot_three_way_confusion_heatmap,
        three_way_confusion_table,
    )
    from src.visualization.plot_model_selection import (
        plot_detection_pareto,
        plot_model_metric_ranking,
        plot_parameter_tendencies,
        plot_three_way_tradeoff,
    )
    from src.visualization.plot_reporting import (
        plot_per_image_performance,
        plot_stage_metric_comparison,
        plot_true_vs_predicted_object_counts,
    )
    from src.visualization.tables import (
        build_candidate_model_table,
        build_per_image_error_table,
    )

    figure_rows: list[dict[str, object]] = []
    global_dir = output_root / "global"
    tables_dir = output_root / "tables"
    global_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    selected_configs = sort_configs(tables["selected_configs"])
    metrics_long = tables["metrics_long"]
    metrics_2way = pd.concat(
        [tables["metrics_2way_object"], tables["metrics_2way_pixel"]],
        ignore_index=True,
        sort=False,
    )
    metrics_3way = tables["metrics_3way_object"]
    object_diag = add_object_count_columns(tables["object_image_diagnostics"])
    pixel_diag = tables["pixel_image_diagnostics"]
    object_3way_diag = tables["object_3way_image_diagnostics"]
    objects = tables["objects"]
    pixels = tables["pixels"]
    objects_3way = tables["objects_3way"]

    compact_config_table = compact_columns(selected_configs, CONFIG_COLUMNS)
    write_table_bundle(compact_config_table, tables_dir / "selected_configurations_compact")
    write_table_bundle(compact_columns(metrics_long, CONFIG_COLUMNS + TWO_WAY_METRICS + THREE_WAY_METRICS), tables_dir / "mixture_metrics_compact")
    write_table_bundle(build_candidate_model_table(metrics_2way), tables_dir / "candidate_model_table_2way")
    write_table_bundle(build_candidate_model_table(metrics_3way), tables_dir / "candidate_model_table_3way")
    write_table_bundle(build_per_image_error_table(object_diag), tables_dir / "object_per_image_errors")
    write_table_bundle(build_per_image_error_table(pixel_diag), tables_dir / "pixel_per_image_errors")
    if not object_3way_diag.empty:
        write_table_bundle(build_per_image_error_table(object_3way_diag), tables_dir / "object_3way_per_image_errors")

    tendencies = build_parameter_tendency_table(selected_configs)
    if not tendencies.empty:
        write_table_bundle(tendencies, tables_dir / "parameter_tendencies")
        fig = plot_parameter_tendencies(
            tendencies,
            family_col="assigned_selection_track",
            top_n=50,
            title="Parameter tendencies among final mixture models",
            show=False,
        )
        save_fig(fig, global_dir / "parameter_tendencies", formats=formats, strict=strict)

    if not metrics_2way.empty:
        fig = plot_detection_pareto(
            metrics_2way,
            fn_col="fn_rate",
            fp_col="fp_rate",
            color_col="balanced_accuracy",
            symbol_col="matrix_family",
            group_col="assigned_selection_track" if "assigned_selection_track" in metrics_2way.columns else None,
            id_col="selected_config_id",
            title="Mixture 2-way detection Pareto by track",
            show=False,
        )
        save_fig(fig, global_dir / "mixture_2way_detection_pareto_all_tracks", formats=formats, strict=strict)

        for metric, ascending in (("fn_rate", True), ("fp_rate", True), ("balanced_accuracy", False)):
            if metric not in metrics_2way.columns:
                continue
            fig = plot_model_metric_ranking(
                metrics_2way,
                metric_col=metric,
                id_col="selected_config_id",
                family_col="assigned_selection_track"
                if "assigned_selection_track" in metrics_2way.columns
                else "matrix_family",
                ascending=ascending,
                top_n=top_n_models,
                title=f"Mixture 2-way model ranking by {metric}",
                show=False,
            )
            save_fig(fig, global_dir / f"mixture_2way_ranking_{metric}", formats=formats, strict=strict)

        for track in TRACK_ORDER:
            sub = metrics_2way[
                metrics_2way.get("assigned_selection_track", metrics_2way.get("selection_track", "")).astype(str).eq(track)
            ] if "assigned_selection_track" in metrics_2way.columns or "selection_track" in metrics_2way.columns else pd.DataFrame()
            if sub.empty:
                continue
            fig = plot_detection_pareto(
                sub,
                fn_col="fn_rate",
                fp_col="fp_rate",
                color_col="balanced_accuracy",
                symbol_col="matrix_family",
                id_col="selected_config_id",
                title=f"Mixture 2-way detection Pareto - {track}",
                show=False,
            )
            save_fig(fig, global_dir / f"mixture_2way_detection_pareto_{track}", formats=formats, strict=strict)

    if not metrics_3way.empty:
        fig = plot_three_way_tradeoff(
            metrics_3way,
            group_col="assigned_selection_track"
            if "assigned_selection_track" in metrics_3way.columns
            else "matrix_family",
            id_col="selected_config_id",
            title="Mixture 3-way trade-off by track",
            show=False,
        )
        save_fig(fig, global_dir / "mixture_3way_tradeoff_all_tracks", formats=formats, strict=strict)

        for metric, ascending in (
            ("target_miss_rate", True),
            ("non_target_false_accept_rate", True),
            ("uncertain_rate", True),
            ("coverage_rate", False),
            ("decided_balanced_accuracy", False),
        ):
            if metric not in metrics_3way.columns:
                continue
            fig = plot_model_metric_ranking(
                metrics_3way,
                metric_col=metric,
                id_col="selected_config_id",
                family_col="assigned_selection_track"
                if "assigned_selection_track" in metrics_3way.columns
                else "matrix_family",
                ascending=ascending,
                top_n=top_n_models,
                title=f"Mixture 3-way model ranking by {metric}",
                show=False,
            )
            save_fig(fig, global_dir / f"mixture_3way_ranking_{metric}", formats=formats, strict=strict)

        for track in TRACK_ORDER:
            if "assigned_selection_track" not in metrics_3way.columns:
                continue
            sub = metrics_3way[metrics_3way["assigned_selection_track"].astype(str).eq(track)]
            if sub.empty:
                continue
            fig = plot_three_way_tradeoff(
                sub,
                group_col="matrix_family" if "matrix_family" in sub.columns else None,
                id_col="selected_config_id",
                title=f"Mixture 3-way trade-off - {track}",
                show=False,
            )
            save_fig(fig, global_dir / f"mixture_3way_tradeoff_{track}", formats=formats, strict=strict)

    if not object_diag.empty:
        fig = plot_per_image_performance(
            object_diag,
            metric_cols=TWO_WAY_METRICS,
            config_col="selected_config_id",
            top_n=40,
            title="Object-level mixture performance by image",
            show=False,
        )
        save_fig(fig, global_dir / "object_per_image_performance", formats=formats, strict=strict)
        if {"n_true_target_objects", "n_predicted_target_objects"}.issubset(object_diag.columns):
            fig = plot_true_vs_predicted_object_counts(
                object_diag,
                config_col="selected_config_id",
                title="True versus predicted target-object counts on mixtures",
                show=False,
            )
            save_fig(fig, global_dir / "true_vs_predicted_object_counts", formats=formats, strict=strict)

    if not pixel_diag.empty:
        fig = plot_per_image_performance(
            pixel_diag,
            metric_cols=("pixel_fn_rate", "pixel_fp_rate", "pixel_balanced_accuracy"),
            sort_metric="pixel_fn_rate",
            config_col="selected_config_id",
            top_n=40,
            title="Pixel-level mixture performance by image",
            show=False,
        )
        save_fig(fig, global_dir / "pixel_per_image_performance", formats=formats, strict=strict)

    if not object_3way_diag.empty:
        fig = plot_per_image_performance(
            object_3way_diag,
            metric_cols=("target_miss_rate", "non_target_false_accept_rate", "uncertain_rate", "coverage_rate"),
            sort_metric="target_miss_rate",
            config_col="selected_config_id",
            top_n=40,
            title="3-way object-level mixture diagnostics by image",
            show=False,
        )
        save_fig(fig, global_dir / "object_3way_per_image_performance", formats=formats, strict=strict)

    binary_object_conf = build_binary_confusion_table(
        objects,
        true_col="true_label_object",
        pred_col="predicted_label_object",
        group_cols=("selected_config_id", "assigned_selection_track"),
        target_class=target_class,
        non_target_label=non_target_label,
    )
    if not binary_object_conf.empty:
        write_table_bundle(binary_object_conf, tables_dir / "binary_object_confusion_long")
        fig = plot_binary_confusion_heatmap(
            binary_object_conf,
            target_class=target_class,
            non_target_label=non_target_label,
            title="Object-level 2-way confusion on mixtures",
            show=False,
        )
        save_fig(fig, global_dir / "object_2way_confusion_all_selected_models", formats=formats, strict=strict)

    binary_pixel_conf = build_binary_confusion_table(
        pixels,
        true_col=f"true_{target_class}_pixel",
        pred_col="predicted_label_pixel",
        group_cols=("selected_config_id", "assigned_selection_track"),
        target_class=target_class,
        non_target_label=non_target_label,
    )
    if not binary_pixel_conf.empty:
        write_table_bundle(binary_pixel_conf, tables_dir / "binary_pixel_confusion_long")
        fig = plot_binary_confusion_heatmap(
            binary_pixel_conf,
            target_class=target_class,
            non_target_label=non_target_label,
            title="Pixel-level 2-way confusion on mixtures",
            show=False,
        )
        save_fig(fig, global_dir / "pixel_2way_confusion_all_selected_models", formats=formats, strict=strict)

    three_way_object_conf = three_way_confusion_table(
        objects_3way,
        true_col=f"true_{target_class}_object",
        decision_col="decision_3way",
        group_cols=("selected_config_id", "assigned_selection_track"),
        target_class=target_class,
        non_target_label=non_target_label,
        uncertain_label=uncertain_label,
    )
    if not three_way_object_conf.empty:
        write_table_bundle(three_way_object_conf, tables_dir / "three_way_object_confusion_long")
        fig = plot_three_way_confusion_heatmap(
            three_way_object_conf,
            target_class=target_class,
            non_target_label=non_target_label,
            uncertain_label=uncertain_label,
            title="Object-level 3-way confusion on mixtures",
            show=False,
        )
        save_fig(fig, global_dir / "object_3way_confusion_all_selected_models", formats=formats, strict=strict)

    if not metrics_long.empty:
        stage_df = build_stage_comparison_table(
            project_root=project_root,
            results_tag=results_tag,
            mixture_metrics_df=metrics_long,
            selected_config_ids=selected_configs["selected_config_id"].astype(str).tolist(),
        )
        if not stage_df.empty:
            keep_cols = [
                col
                for col in (
                    "selected_config_id",
                    "assigned_selection_track",
                    "selection_track",
                    "matrix_family",
                    "decision_mode",
                    "metric_level",
                    "evaluation_stage",
                    *TWO_WAY_METRICS,
                    *THREE_WAY_METRICS,
                )
                if col in stage_df.columns
            ]
            write_table_bundle(stage_df[keep_cols], tables_dir / "stage_metric_comparison_compact")
            for metric in ("fn_rate", "fp_rate", "balanced_accuracy", "target_miss_rate", "non_target_false_accept_rate", "uncertain_rate", "coverage_rate"):
                if metric not in stage_df.columns:
                    continue
                plot_df = stage_df.dropna(subset=[metric]).copy()
                if plot_df.empty:
                    continue
                if len(plot_df["selected_config_id"].astype(str).unique()) > top_n_models:
                    selected_ids = selected_configs["selected_config_id"].astype(str).head(top_n_models).tolist()
                    plot_df = plot_df[plot_df["selected_config_id"].astype(str).isin(selected_ids)].copy()
                fig = plot_stage_metric_comparison(
                    plot_df,
                    metric_col=metric,
                    stage_col="evaluation_stage",
                    config_col="selected_config_id",
                    title=f"{metric} across validation, robustness, pure test and mixture",
                    show=False,
                )
                save_fig(fig, global_dir / f"stage_comparison_{metric}", formats=formats, strict=strict)

    figure_rows.append({"section": "global", "path": str(global_dir)})
    return figure_rows


def maybe_load_database(db_h5: Path, *, skip_spatial: bool):
    if skip_spatial:
        return {}, {}
    if not db_h5.exists():
        print(f"[INFO] HDF5 database not found, skipping spatial figures: {db_h5}")
        return {}, {}
    from src.io.database_h5 import load_nir_uco_h5

    object_db, image_db = load_nir_uco_h5(db_h5, reconstruct_heavy_object_arrays=True)
    return image_db, object_db


def make_config_figures(
    *,
    configs_df: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    output_root: Path,
    formats: Sequence[str],
    image_db: Mapping,
    object_db: Mapping,
    target_class: str,
    non_target_label: str,
    uncertain_label: str,
    n_images: int,
    image_selection_mode: str,
    max_pixels_qt2: int,
    strict: bool,
    skip_spatial: bool,
) -> pd.DataFrame:
    """Create per-configuration maps, confusion heatmaps and SIMCA diagnostics."""
    from src.visualization.plot_decision import (
        plot_binary_confusion_heatmap,
        plot_object_decision_map,
        plot_object_error_overlay,
        plot_pixel_error_overlay,
        plot_pixel_prediction_overlay,
        plot_pixel_three_way_decision_overlay,
        plot_three_way_confusion_heatmap,
        three_way_confusion_table,
    )
    from src.visualization.plot_reporting import plot_mixture_diagnostic_panel
    from src.visualization.plot_simca import plot_simca_q_t2_dataframe

    objects = add_plot_labels(
        tables["objects"],
        target_class=target_class,
        non_target_label=non_target_label,
        uncertain_label=uncertain_label,
    )
    pixels = add_plot_labels(
        tables["pixels"],
        target_class=target_class,
        non_target_label=non_target_label,
        uncertain_label=uncertain_label,
    )
    objects_3way = add_plot_labels(
        tables["objects_3way"],
        target_class=target_class,
        non_target_label=non_target_label,
        uncertain_label=uncertain_label,
    )
    pixel_3way = build_pixel_3way_view(pixels, objects_3way)
    pixel_3way = add_plot_labels(
        pixel_3way,
        target_class=target_class,
        non_target_label=non_target_label,
        uncertain_label=uncertain_label,
    )

    object_diag = add_object_count_columns(tables["object_image_diagnostics"])
    pixel_diag = tables["pixel_image_diagnostics"]
    object_3way_diag = tables["object_3way_image_diagnostics"]

    summary_rows: list[dict[str, object]] = []
    two_way_categories = {
        1: {"label": non_target_label, "color": "royalblue"},
        2: {"label": target_class, "color": "limegreen"},
    }
    three_way_categories = {
        1: {"label": non_target_label, "color": "royalblue"},
        2: {"label": uncertain_label, "color": "purple"},
        3: {"label": target_class, "color": "limegreen"},
    }
    two_way_order = [non_target_label, target_class]
    two_way_colors = {non_target_label: "royalblue", target_class: "limegreen"}
    three_way_order = [non_target_label, uncertain_label, target_class]
    three_way_colors = {non_target_label: "royalblue", uncertain_label: "purple", target_class: "limegreen"}

    for _, cfg in configs_df.iterrows():
        config_id = str(cfg["selected_config_id"])
        config_dir = output_root / "by_config" / config_folder_name(cfg)
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.to_frame().T.to_csv(config_dir / "config_info.csv", index=False)

        obj_cfg = filter_config(objects, config_id)
        pix_cfg = filter_config(pixels, config_id)
        obj3_cfg = filter_config(objects_3way, config_id)
        pix3_cfg = filter_config(pixel_3way, config_id)

        if str(cfg.get("decision_mode", "")).lower() == "3way":
            image_metric_source = object_3way_diag
        elif str(cfg.get("metric_level", "")).lower() == "pixel":
            image_metric_source = pixel_diag
        else:
            image_metric_source = object_diag
        selected_images = choose_images(
            image_metric_source,
            config_id=config_id,
            n_images=n_images,
            mode=image_selection_mode,
        )
        if selected_images.empty and not obj_cfg.empty and "source_image" in obj_cfg.columns:
            selected_images = obj_cfg[["source_image"]].drop_duplicates().head(n_images).copy()
            selected_images["image_selection_reason"] = "fallback_first_images"
        selected_images.to_csv(config_dir / "selected_images.csv", index=False)
        image_keys = selected_images["source_image"].astype(str).tolist() if "source_image" in selected_images.columns else []

        print(f"\n[CONFIG] {config_id} -> {config_dir}")
        print("  images:", ", ".join(image_keys) if image_keys else "none")

        binary_obj_conf = build_binary_confusion_table(
            obj_cfg,
            true_col="true_label_object",
            pred_col="predicted_label_object",
            target_class=target_class,
            non_target_label=non_target_label,
        )
        if not binary_obj_conf.empty:
            write_table_bundle(binary_obj_conf, config_dir / "tables" / "object_2way_confusion")
            fig = plot_binary_confusion_heatmap(
                binary_obj_conf,
                target_class=target_class,
                non_target_label=non_target_label,
                title=f"Object-level 2-way confusion - {config_id}",
                show=False,
            )
            save_fig(fig, config_dir / "confusion" / "object_2way_confusion", formats=formats, strict=strict)

        binary_pix_conf = build_binary_confusion_table(
            pix_cfg,
            true_col=f"true_{target_class}_pixel",
            pred_col="predicted_label_pixel",
            target_class=target_class,
            non_target_label=non_target_label,
        )
        if not binary_pix_conf.empty:
            write_table_bundle(binary_pix_conf, config_dir / "tables" / "pixel_2way_confusion")
            fig = plot_binary_confusion_heatmap(
                binary_pix_conf,
                target_class=target_class,
                non_target_label=non_target_label,
                title=f"Pixel-level 2-way confusion - {config_id}",
                show=False,
            )
            save_fig(fig, config_dir / "confusion" / "pixel_2way_confusion", formats=formats, strict=strict)

        three_way_obj_conf = three_way_confusion_table(
            obj3_cfg,
            true_col=f"true_{target_class}_object",
            decision_col="decision_3way",
            target_class=target_class,
            non_target_label=non_target_label,
            uncertain_label=uncertain_label,
        )
        if not three_way_obj_conf.empty:
            write_table_bundle(three_way_obj_conf, config_dir / "tables" / "object_3way_confusion")
            fig = plot_three_way_confusion_heatmap(
                three_way_obj_conf,
                target_class=target_class,
                non_target_label=non_target_label,
                uncertain_label=uncertain_label,
                title=f"Object-level 3-way confusion - {config_id}",
                show=False,
            )
            save_fig(fig, config_dir / "confusion" / "object_3way_confusion", formats=formats, strict=strict)

        try:
            label_col = resolve_label_column(obj_cfg, ("predicted_label_object_plot", "predicted_label_object"))
            if label_col is not None and not obj_cfg.empty:
                fig = plot_simca_q_t2_dataframe(
                    obj_cfg,
                    level="object",
                    label_col=label_col,
                    title=f"Object-level SIMCA H/Q diagnostics - {config_id}",
                    category_order=two_way_order,
                    color_map=two_way_colors,
                    show=False,
                )
                save_fig(fig, config_dir / "diagnostics" / "object_q_h_diagnostics_2way", formats=formats, strict=strict)
        except Exception as exc:
            print(f"  [WARNING] Object Q/H diagnostic skipped for {config_id}: {exc!r}")

        try:
            label_col = resolve_label_column(pix_cfg, ("predicted_label_pixel_plot", "predicted_label_pixel"))
            if label_col is not None and not pix_cfg.empty:
                pix_plot = sample_for_qt2(pix_cfg, label_col=label_col, max_rows=max_pixels_qt2)
                fig = plot_simca_q_t2_dataframe(
                    pix_plot,
                    level="pixel",
                    label_col=label_col,
                    title=f"Pixel-level SIMCA H/Q diagnostics - {config_id}",
                    category_order=two_way_order,
                    color_map=two_way_colors,
                    show=False,
                )
                save_fig(fig, config_dir / "diagnostics" / "pixel_q_h_diagnostics_2way", formats=formats, strict=strict)
        except Exception as exc:
            print(f"  [WARNING] Pixel Q/H diagnostic skipped for {config_id}: {exc!r}")

        try:
            label_col = resolve_label_column(obj3_cfg, ("decision_3way_plot", "decision_3way"))
            if label_col is not None and not obj3_cfg.empty:
                fig = plot_simca_q_t2_dataframe(
                    obj3_cfg,
                    level="object",
                    label_col=label_col,
                    confidence_col="three_way_confidence",
                    title=f"Object-level SIMCA H/Q diagnostics, 3-way decision - {config_id}",
                    category_order=three_way_order,
                    color_map=three_way_colors,
                    show=False,
                )
                save_fig(fig, config_dir / "diagnostics" / "object_q_h_diagnostics_3way", formats=formats, strict=strict)
        except Exception as exc:
            print(f"  [WARNING] Object 3-way Q/H diagnostic skipped for {config_id}: {exc!r}")

        if not skip_spatial and image_db and object_db:
            for image_key in image_keys:
                if image_key not in image_db:
                    print(f"  [WARNING] Image key not found in database, skipping maps: {image_key}")
                    continue
                try:
                    fig = plot_mixture_diagnostic_panel(
                        image_key=image_key,
                        image_db=image_db,
                        object_db=object_db,
                        object_df=obj_cfg,
                        pixel_df=pix_cfg,
                        target_class=target_class,
                        title=f"Mixture diagnostic panel - {config_id} - {image_key}",
                        show=False,
                    )
                    save_fig(fig, config_dir / "spatial" / f"{image_key}_diagnostic_panel", formats=formats, strict=strict)
                except Exception as exc:
                    print(f"  [WARNING] Mixture diagnostic panel skipped for {config_id}/{image_key}: {exc!r}")

                try:
                    decision_col = resolve_label_column(obj_cfg, ("predicted_label_object_plot", "predicted_label_object"))
                    if decision_col is not None:
                        fig = plot_object_decision_map(
                            image_db=image_db,
                            object_db=object_db,
                            results_df=obj_cfg,
                            image_key=image_key,
                            decision_col=decision_col,
                            decision_to_code={non_target_label: 1, target_class: 2},
                            categories=two_way_categories,
                            title=f"Object-level 2-way decisions - {config_id} - {image_key}",
                            show=False,
                        )
                        save_fig(fig, config_dir / "spatial" / f"{image_key}_object_decisions_2way", formats=formats, strict=strict)
                except Exception as exc:
                    print(f"  [WARNING] Object decision map skipped for {config_id}/{image_key}: {exc!r}")

                try:
                    fig = plot_object_error_overlay(
                        image_key=image_key,
                        image_db=image_db,
                        object_db=object_db,
                        object_df=obj_cfg,
                        target_class=target_class,
                        title=f"Object-level 2-way errors - {config_id} - {image_key}",
                        show=False,
                    )
                    save_fig(fig, config_dir / "spatial" / f"{image_key}_object_errors_2way", formats=formats, strict=strict)
                except Exception as exc:
                    print(f"  [WARNING] Object error map skipped for {config_id}/{image_key}: {exc!r}")

                if not pix_cfg.empty:
                    try:
                        pred_col = resolve_label_column(pix_cfg, ("predicted_label_pixel", f"predicted_{target_class}_pixel"))
                        fig = plot_pixel_prediction_overlay(
                            image_key=image_key,
                            image_db=image_db,
                            pixel_df=pix_cfg,
                            target_class=target_class,
                            pred_col=pred_col,
                            title=f"Pixel-level 2-way predictions - {config_id} - {image_key}",
                            show=False,
                        )
                        save_fig(fig, config_dir / "spatial" / f"{image_key}_pixel_predictions_2way", formats=formats, strict=strict)
                    except Exception as exc:
                        print(f"  [WARNING] Pixel prediction map skipped for {config_id}/{image_key}: {exc!r}")

                    try:
                        fig = plot_pixel_error_overlay(
                            image_key=image_key,
                            image_db=image_db,
                            pixel_df=pix_cfg,
                            target_class=target_class,
                            title=f"Pixel-level 2-way errors - {config_id} - {image_key}",
                            show=False,
                        )
                        save_fig(fig, config_dir / "spatial" / f"{image_key}_pixel_errors_2way", formats=formats, strict=strict)
                    except Exception as exc:
                        print(f"  [WARNING] Pixel error map skipped for {config_id}/{image_key}: {exc!r}")

                if not obj3_cfg.empty:
                    try:
                        decision_col = resolve_label_column(obj3_cfg, ("decision_3way_plot", "decision_3way"))
                        if decision_col is not None:
                            fig = plot_object_decision_map(
                                image_db=image_db,
                                object_db=object_db,
                                results_df=obj3_cfg,
                                image_key=image_key,
                                decision_col=decision_col,
                                decision_to_code={
                                    non_target_label: 1,
                                    uncertain_label: 2,
                                    target_class: 3,
                                },
                                categories=three_way_categories,
                                title=f"Object-level 3-way decisions - {config_id} - {image_key}",
                                show=False,
                            )
                            save_fig(fig, config_dir / "spatial" / f"{image_key}_object_decisions_3way", formats=formats, strict=strict)
                    except Exception as exc:
                        print(f"  [WARNING] Object 3-way decision map skipped for {config_id}/{image_key}: {exc!r}")

                if not pix3_cfg.empty and "decision_3way" in pix3_cfg.columns:
                    try:
                        decision_col = "decision_3way_plot" if "decision_3way_plot" in pix3_cfg.columns else "decision_3way"
                        fig = plot_pixel_three_way_decision_overlay(
                            image_key=image_key,
                            image_db=image_db,
                            pixel_df=pix3_cfg,
                            decision_col=decision_col,
                            target_class=target_class,
                            non_target_label=non_target_label,
                            uncertain_label=uncertain_label,
                            title=f"Pixel view of 3-way decisions - {config_id} - {image_key}",
                            show=False,
                        )
                        save_fig(fig, config_dir / "spatial" / f"{image_key}_pixel_decisions_3way", formats=formats, strict=strict)
                    except Exception as exc:
                        print(f"  [WARNING] Pixel 3-way decision map skipped for {config_id}/{image_key}: {exc!r}")

        summary_rows.append(
            {
                "selected_config_id": config_id,
                "assigned_selection_track": cfg.get("assigned_selection_track", cfg.get("selection_track", "")),
                "config_dir": str(config_dir),
                "n_selected_images": len(image_keys),
                "selected_images": ";".join(image_keys),
            }
        )

    return pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mixture report figures from notebook-07 saved outputs."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--results-tag", type=str, default="non_noisy_all")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--db-h5", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", default=["html"], help="Figure formats: html, png, svg, pdf, json.")
    parser.add_argument("--strict-static-export", action="store_true", help="Fail when a requested static export cannot be written.")
    parser.add_argument("--target-class", type=str, default="peanut")
    parser.add_argument("--non-target-label", type=str, default="almond")
    parser.add_argument("--uncertain-label", type=str, default="uncertain")
    parser.add_argument("--top-n-models", type=int, default=20)
    parser.add_argument("--n-configs-per-track", type=int, default=3)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--image-selection-mode", choices=["hardest", "easiest", "both"], default="hardest")
    parser.add_argument("--max-pixels-qt2", type=int, default=8000)
    parser.add_argument("--skip-spatial", action="store_true", help="Skip HDF5 loading and spatial maps.")
    parser.add_argument("--only-global", action="store_true", help="Generate only global figures and compact report tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(args.project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    results_dir = (
        args.results_dir
        if args.results_dir is not None
        else project_root / "results" / f"07_simca_mixture_application_{args.results_tag}"
    )
    output_root = args.output_root if args.output_root is not None else results_dir / "figures"
    db_h5 = args.db_h5 if args.db_h5 is not None else project_root / "HSI Data" / "processed" / "nir_uco_database.h5"

    print("PROJECT_ROOT:", project_root)
    print("RESULTS_DIR:", results_dir)
    print("OUTPUT_ROOT:", output_root)
    print("FORMATS:", ", ".join(args.formats))

    paths = build_mixture_paths(results_dir)
    require_existing(
        [
            paths.selected_configs,
            paths.metrics_long,
            paths.object_image_diagnostics,
            paths.pixel_image_diagnostics,
            paths.objects,
        ]
    )
    output_root.mkdir(parents=True, exist_ok=True)

    tables = {
        "selected_configs": read_table(paths.selected_configs, required=True),
        "metrics_long": read_table(paths.metrics_long, required=True),
        "metrics_2way_object": read_table(paths.metrics_2way_object),
        "metrics_2way_pixel": read_table(paths.metrics_2way_pixel),
        "metrics_3way_object": read_table(paths.metrics_3way_object),
        "object_image_diagnostics": read_table(paths.object_image_diagnostics, required=True),
        "pixel_image_diagnostics": read_table(paths.pixel_image_diagnostics, required=True),
        "object_3way_image_diagnostics": read_table(paths.object_3way_image_diagnostics),
        "pixel_errors_by_image": read_table(paths.pixel_errors_by_image),
        "objects": read_table(paths.objects, required=True),
        "pixels": read_table(paths.pixels),
        "objects_3way": read_table(paths.objects_3way),
        "summary": read_table(paths.summary),
        "guardrails": read_table(paths.guardrails),
        "protocol": read_table(paths.protocol),
        "errors": read_table(paths.errors),
    }

    print("\nLoaded tables:")
    for name, table in tables.items():
        print(f"  {name:30s} {table.shape}")

    make_global_figures(
        tables=tables,
        output_root=output_root,
        formats=args.formats,
        top_n_models=args.top_n_models,
        strict=args.strict_static_export,
        project_root=project_root,
        results_tag=args.results_tag,
        target_class=args.target_class,
        non_target_label=args.non_target_label,
        uncertain_label=args.uncertain_label,
    )

    selected_configs = sort_configs(tables["selected_configs"])
    report_configs = select_report_configs(
        selected_configs,
        n_configs_per_track=args.n_configs_per_track,
        max_configs=args.max_configs,
    )
    write_table_bundle(compact_columns(report_configs, CONFIG_COLUMNS), output_root / "tables" / "report_configuration_panel")

    if args.only_global:
        summary_df = pd.DataFrame(
            [
                {
                    "section": "global_only",
                    "output_root": str(output_root),
                    "n_report_configs": len(report_configs),
                }
            ]
        )
    else:
        image_db, object_db = maybe_load_database(db_h5, skip_spatial=args.skip_spatial)
        summary_df = make_config_figures(
            configs_df=report_configs,
            tables=tables,
            output_root=output_root,
            formats=args.formats,
            image_db=image_db,
            object_db=object_db,
            target_class=args.target_class,
            non_target_label=args.non_target_label,
            uncertain_label=args.uncertain_label,
            n_images=args.n_images,
            image_selection_mode=args.image_selection_mode,
            max_pixels_qt2=args.max_pixels_qt2,
            strict=args.strict_static_export,
            skip_spatial=args.skip_spatial,
        )

    summary_path = output_root / "figure_generation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nDone.")
    print("Figure root:", output_root)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
