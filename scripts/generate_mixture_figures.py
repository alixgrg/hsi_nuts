"""
Generate SIMCA mixture figures from saved projection tables, organized by configuration.

This script is intended to be run after notebook 05 has saved the projection
parquet files. It does not refit any model and does not recalibrate thresholds.

Main features
-------------
- One output subfolder per selected configuration.
- Uses the plotting functions from src.visualization modules.
- Selects images with the best FN / target-miss ratios by default.
- Limits the number of selected images with only one true peanut object.
- Tightens spatial figure framing to avoid extra bands around projected images.

Example
-------
python scripts/generate_mixture_figures.py ^
  --project-root "C:/Users/alixg/OneDrive - Université Paris-Dauphine/hsi_nuts" ^
  --results-tag non_noisy_all ^
  --save-html ^
  --save-png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists():
            return candidate
    raise RuntimeError(
        "Could not find project root. Run from the project root, "
        "or pass --project-root."
    )


def safe_filename(text: str, max_len: int = 170) -> str:
    text = str(text)
    text = re.sub(r"[<>:\"/\\|?*]", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text[:max_len].strip("_")


def read_parquet_if_exists(path: Path, required: bool = False) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    if required:
        raise FileNotFoundError(f"Required parquet not found: {path}")
    print(f"[INFO] Missing optional parquet: {path}")
    return pd.DataFrame()


def first_existing_parquet(paths: Iterable[Path], required: bool = False) -> tuple[pd.DataFrame, Path | None]:
    for path in paths:
        if path.exists():
            return pd.read_parquet(path), path
    if required:
        raise FileNotFoundError("None of these parquet files exists:\n" + "\n".join(map(str, paths)))
    return pd.DataFrame(), None


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {name}: {missing}")


def value_from_row(row: pd.Series, cols: list[str], default: str = "") -> str:
    for col in cols:
        if col in row.index and pd.notna(row[col]) and str(row[col]) != "":
            return str(row[col])
    return default


def make_config_folder_name(row: pd.Series) -> str:
    config_id = value_from_row(row, ["selected_config_id"], "config")
    family = value_from_row(row, ["matrix_family"], "family")
    matrix = value_from_row(row, ["training_matrix_id", "matrix_method"], "matrix")
    preproc = value_from_row(row, ["preprocessing"], "preproc")
    rule = value_from_row(row, ["selected_rule_name", "rule_variant", "rule_for_refit", "rule"], "rule")
    return safe_filename(f"{config_id}__{family}__{matrix}__{preproc}__{rule}", max_len=190)


def save_plotly_figure(
    fig: go.Figure | None,
    stem: str,
    config_dir: Path,
    save_html: bool = True,
    save_png: bool = True,
    png_width: int = 1400,
    png_height: int = 1000,
    png_scale: int = 2,
) -> None:
    if fig is None:
        print(f"[WARNING] Figure is None, not saved: {stem}")
        return

    stem = safe_filename(stem)

    if save_html:
        html_dir = config_dir / "html"
        html_dir.mkdir(parents=True, exist_ok=True)
        html_path = html_dir / f"{stem}.html"
        fig.write_html(html_path)
        print("  HTML:", html_path)

    if save_png:
        png_dir = config_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        png_path = png_dir / f"{stem}.png"
        try:
            fig.write_image(
                png_path,
                width=int(png_width),
                height=int(png_height),
                scale=int(png_scale),
            )
            print("  PNG :", png_path)
        except Exception as exc:
            print(f"  [WARNING] Could not save PNG for {stem}: {exc!r}")
            print("            Install python-kaleido and Chrome/Chromium for PNG export.")


# -----------------------------------------------------------------------------
# Configuration / image selection helpers
# -----------------------------------------------------------------------------


def choose_diagnostic_configs(
    reference_df: pd.DataFrame,
    n_per_family: int = 2,
    max_configs: int | None = None,
) -> pd.DataFrame:
    if reference_df is None or len(reference_df) == 0:
        return pd.DataFrame()

    df = reference_df.copy()
    df["selected_config_id"] = df["selected_config_id"].astype(str)

    if "frozen_reference_rank" not in df.columns:
        df["frozen_reference_rank"] = np.arange(1, len(df) + 1)

    group_cols = [c for c in ["matrix_family", "candidate_source"] if c in df.columns]

    if group_cols:
        out = (
            df.sort_values("frozen_reference_rank")
            .groupby(group_cols, group_keys=False, dropna=False)
            .head(int(n_per_family))
            .reset_index(drop=True)
        )
    else:
        out = df.sort_values("frozen_reference_rank").head(int(n_per_family)).copy()

    if max_configs is not None:
        out = out.head(int(max_configs)).copy()

    return out.reset_index(drop=True)


def infer_true_target_object_col(object_df: pd.DataFrame, target_class: str) -> str | None:
    candidates = [
        f"true_{target_class}_object",
        "true_target_object",
        "true_object",
    ]
    for col in candidates:
        if col in object_df.columns:
            return col
    return None


def count_true_target_objects_by_image(
    object_df: pd.DataFrame,
    target_class: str = "peanut",
) -> dict[str, int]:
    """Return {source_image: number of true target objects} for one config table."""
    if object_df is None or len(object_df) == 0 or "source_image" not in object_df.columns:
        return {}

    true_col = infer_true_target_object_col(object_df, target_class=target_class)
    if true_col is None:
        return {}

    d = object_df.copy()
    d = d[d[true_col].notna()].copy()
    if len(d) == 0:
        return {}

    if "object_id" in d.columns:
        d = d.drop_duplicates(["source_image", "object_id"])

    counts = (
        d.assign(_is_target=d[true_col].astype(bool))
        .groupby("source_image", dropna=False)["_is_target"]
        .sum()
        .astype(int)
    )
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def add_target_counts_to_by_image(
    by_image_df: pd.DataFrame,
    object_df_config: pd.DataFrame,
    target_class: str,
) -> pd.DataFrame:
    out = by_image_df.copy()
    if "source_image" not in out.columns:
        return out

    count_map = count_true_target_objects_by_image(
        object_df_config,
        target_class=target_class,
    )

    if count_map:
        out["n_true_target_objects"] = out["source_image"].astype(str).map(count_map).astype("float")
    elif "n_target" in out.columns:
        out["n_true_target_objects"] = pd.to_numeric(out["n_target"], errors="coerce")
    else:
        out["n_true_target_objects"] = np.nan

    return out


def sort_by_metric_preferences(
    df: pd.DataFrame,
    metric_preferences: list[tuple[str, str]],
    mode: str = "best",
) -> pd.DataFrame:
    """
    metric_preferences: [(metric_col, 'low' or 'high'), ...]
    mode='best' keeps low metrics low and high metrics high.
    mode='worst' reverses directions.
    """
    d = df.copy()
    sort_cols: list[str] = []
    ascending: list[bool] = []

    for col, better in metric_preferences:
        if col not in d.columns:
            continue
        d[col] = pd.to_numeric(d[col], errors="coerce")
        sort_cols.append(col)
        asc = better == "low"
        if mode == "worst":
            asc = not asc
        ascending.append(asc)

    if sort_cols:
        d = d.sort_values(sort_cols, ascending=ascending, na_position="last")

    return d.reset_index(drop=True)


def choose_images_for_config(
    config_id: str,
    by_image_df: pd.DataFrame,
    object_df_config: pd.DataFrame,
    metric_preferences: list[tuple[str, str]],
    n_images: int = 3,
    mode: str = "best",
    max_single_target_images: int = 1,
    target_class: str = "peanut",
) -> pd.DataFrame:
    """
    Select images for one configuration.

    Default behavior:
    - rank by best FN / miss ratio first;
    - allow at most one image with <= 1 true target object;
    - return fewer than n_images if the constraint cannot be satisfied.
    """
    if object_df_config is None:
        object_df_config = pd.DataFrame()

    if by_image_df is not None and len(by_image_df) > 0 and "selected_config_id" in by_image_df.columns:
        d = by_image_df[by_image_df["selected_config_id"].astype(str).eq(str(config_id))].copy()
    else:
        d = pd.DataFrame()

    if len(d) == 0:
        if object_df_config is not None and len(object_df_config) > 0 and "source_image" in object_df_config.columns:
            d = object_df_config[["source_image"]].drop_duplicates().copy()
        else:
            return pd.DataFrame(columns=["source_image", "selection_reason"])

    if "source_image" not in d.columns:
        return pd.DataFrame(columns=["source_image", "selection_reason"])

    d = d[d["source_image"].notna()].copy()
    d["source_image"] = d["source_image"].astype(str)
    d = add_target_counts_to_by_image(d, object_df_config, target_class=target_class)
    d = sort_by_metric_preferences(d, metric_preferences=metric_preferences, mode=mode)
    d = d.drop_duplicates("source_image").reset_index(drop=True)

    selected_rows = []
    n_single = 0

    for _, row in d.iterrows():
        n_target = row.get("n_true_target_objects", np.nan)
        is_single_target = pd.notna(n_target) and float(n_target) <= 1.0

        if is_single_target and n_single >= int(max_single_target_images):
            continue

        if is_single_target:
            n_single += 1

        row = row.copy()
        row["selection_rank"] = len(selected_rows) + 1
        row["selection_reason"] = f"{mode}_fn_with_single_target_limit"
        selected_rows.append(row)

        if len(selected_rows) >= int(n_images):
            break

    out = pd.DataFrame(selected_rows)
    if len(out) < int(n_images):
        print(
            f"  [INFO] Only {len(out)}/{n_images} images selected for {config_id} "
            f"after applying max_single_target_images={max_single_target_images}."
        )

    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Figure framing helpers
# -----------------------------------------------------------------------------


def image_shape(image_db: dict, image_key: str, base: str = "image_ref") -> tuple[int, int]:
    img = image_db[image_key]
    arr = img.get(base, img.get("image_ref", img.get("labels")))
    if arr is None:
        raise KeyError(f"Image {image_key!r} has no {base!r}, image_ref or labels.")
    arr = np.asarray(arr)
    return int(arr.shape[0]), int(arr.shape[1])


def tight_frame_spatial_figure(
    fig: go.Figure,
    image_db: dict,
    image_key: str,
    base: str = "image_ref",
    plot_height_px: int = 850,
    colorbar_extra_width_px: int = 260,
    top_margin_px: int = 80,
    bottom_margin_px: int = 30,
    left_margin_px: int = 40,
    right_margin_px: int = 170,
    equal_aspect: bool = True,
) -> go.Figure:
    """
    Tighten spatial image figures so there are no wide bands around the image.

    The plotting modules keep a square aspect ratio, which can create large
    apparent margins if the figure width/height is not adapted to the image.
    This helper uses the image shape to set the axis ranges and figure size.
    """
    h, w = image_shape(image_db, image_key=image_key, base=base)

    # Figure size: preserve the image aspect in the plotting area, then add
    # margins and colorbar space.
    plot_area_height = int(plot_height_px)
    plot_area_width = max(400, int(round(plot_area_height * (w / max(h, 1)))))
    fig_width = plot_area_width + int(left_margin_px) + int(right_margin_px) + int(colorbar_extra_width_px)
    fig_height = plot_area_height + int(top_margin_px) + int(bottom_margin_px)

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        autosize=False,
        margin=dict(l=left_margin_px, r=right_margin_px, t=top_margin_px, b=bottom_margin_px),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(
        range=[-0.5, w - 0.5],
        showgrid=False,
        zeroline=False,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[h - 0.5, -0.5],
        showgrid=False,
        zeroline=False,
        constrain="domain",
    )

    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig.update_yaxes(scaleanchor=None)

    return fig


def normalize_binary_label(value, target_class: str = "peanut", non_target_label: str = "almond") -> str:
    if pd.isna(value):
        return "missing"

    val = str(value).lower()

    if val in {
        str(target_class).lower(),
        "target",
        "true",
        "1",
        "peanut",
        "peanut_only",
    }:
        return str(target_class)

    if val in {
        str(non_target_label).lower(),
        "non_target",
        "non_peanut",
        "false",
        "0",
        "almond",
        "almond_only",
    }:
        return str(non_target_label)

    return str(value)


def normalize_three_way_label(
    value,
    target_class: str = "peanut",
    non_target_label: str = "almond",
    uncertain_label: str = "uncertain",
) -> str:
    if pd.isna(value):
        return "missing"

    val = str(value).lower()

    if val in {
        str(target_class).lower(),
        "target",
        "peanut",
        "peanut_only",
    }:
        return str(target_class)

    if val in {
        str(non_target_label).lower(),
        "non_target",
        "non_peanut",
        "almond",
        "almond_only",
    }:
        return str(non_target_label)

    if val in {
        str(uncertain_label).lower(),
        "uncertain",
        "ambiguous",
    }:
        return str(uncertain_label)

    return str(value)


def add_plot_label_columns(
    df: pd.DataFrame,
    target_class: str = "peanut",
    non_target_label: str = "almond",
    uncertain_label: str = "uncertain",
) -> pd.DataFrame:
    """
    Add clean plotting labels without modifying the original decision columns.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()

    if "predicted_label_object" in out.columns:
        out["predicted_label_object_plot"] = out["predicted_label_object"].apply(
            lambda x: normalize_binary_label(
                x,
                target_class=target_class,
                non_target_label=non_target_label,
            )
        )

    if "predicted_label_pixel" in out.columns:
        out["predicted_label_pixel_plot"] = out["predicted_label_pixel"].apply(
            lambda x: normalize_binary_label(
                x,
                target_class=target_class,
                non_target_label=non_target_label,
            )
        )

    if "decision_3way" in out.columns:
        out["decision_3way_plot"] = out["decision_3way"].apply(
            lambda x: normalize_three_way_label(
                x,
                target_class=target_class,
                non_target_label=non_target_label,
                uncertain_label=uncertain_label,
            )
        )

    return out


def save_spatial_figure(
    fig: go.Figure | None,
    stem: str,
    image_db: dict,
    image_key: str,
    config_dir: Path,
    save_html: bool,
    save_png: bool,
    png_width: int,
    png_height: int,
    png_scale: int,
    plot_height_px: int,
    equal_aspect: bool = True,
) -> None:
    if fig is not None:
        fig = tight_frame_spatial_figure(
            fig,
            image_db=image_db,
            image_key=image_key,
            plot_height_px=plot_height_px,
            equal_aspect=equal_aspect,
        )
        # For spatial figures, use the computed figure dimensions rather than
        # the generic PNG_WIDTH/PNG_HEIGHT arguments.
        png_width = int(fig.layout.width) if fig.layout.width else int(png_width)
        png_height = int(fig.layout.height) if fig.layout.height else int(png_height)
    save_plotly_figure(
        fig,
        stem=stem,
        config_dir=config_dir,
        save_html=save_html,
        save_png=save_png,
        png_width=png_width,
        png_height=png_height,
        png_scale=png_scale,
    )


# -----------------------------------------------------------------------------
# Confusion plotting helpers using module functions
# -----------------------------------------------------------------------------


def detect_confusion_columns(confusion_df: pd.DataFrame, mode: str) -> tuple[str | None, str | None]:
    if confusion_df is None or len(confusion_df) == 0:
        return None, None

    if mode == "3way":
        true_candidates = ["true_label_3way", "true_label_binary", "true_label_2way", "true_label"]
        pred_candidates = ["decision_3way", "predicted_label_3way", "predicted_label_binary", "predicted_label_2way"]
    else:
        true_candidates = ["true_label_2way", "true_label_binary", "true_label_3way", "true_label"]
        pred_candidates = ["predicted_label_2way", "predicted_label_binary", "predicted_label_object", "decision_2way"]

    true_name = next((c for c in true_candidates if c in confusion_df.columns), None)
    pred_name = next((c for c in pred_candidates if c in confusion_df.columns), None)
    return true_name, pred_name


def plot_saved_confusion(
    confusion_df: pd.DataFrame,
    config_id: str,
    mode: str,
    title: str,
    plot_confusion_heatmap_from_long,
) -> go.Figure | None:
    if confusion_df is None or len(confusion_df) == 0:
        return None
    if "selected_config_id" in confusion_df.columns:
        d = confusion_df[confusion_df["selected_config_id"].astype(str).eq(str(config_id))].copy()
    else:
        d = confusion_df.copy()
    if len(d) == 0:
        return None

    true_name, pred_name = detect_confusion_columns(d, mode=mode)
    if true_name is None or pred_name is None:
        print(f"  [WARNING] Could not detect confusion columns for {mode}: {list(d.columns)}")
        return None

    try:
        return plot_confusion_heatmap_from_long(
            confusion_df=d,
            true_col_name=true_name,
            decision_col_name=pred_name,
            title=title,
            show=False,
        )
    except TypeError:
        # If a local module has an older signature, use a compact direct Plotly fallback.
        pivot = d.pivot_table(index=true_name, columns=pred_name, values="n", aggfunc="sum", fill_value=0)
        row_total = pivot.sum(axis=1).replace(0, np.nan)
        rate = pivot.div(row_total, axis=0).fillna(0.0)
        text = [[f"{int(pivot.loc[i, j])}<br>{rate.loc[i, j]:.1%}" for j in pivot.columns] for i in pivot.index]
        fig = go.Figure(
            data=go.Heatmap(
                z=rate.to_numpy(dtype=float),
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                text=text,
                texttemplate="%{text}",
                colorscale="Blues",
                colorbar=dict(title="row rate"),
            )
        )
        fig.update_layout(title=title, xaxis_title="Decision", yaxis_title="True label", width=750, height=550)
        return fig


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2-way and 3-way SIMCA mixture figures from saved parquet projections."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--results-tag", type=str, default="non_noisy_all")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--db-h5", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)

    parser.add_argument("--target-class", type=str, default="peanut")
    parser.add_argument("--non-target-label", type=str, default="almond")
    parser.add_argument("--uncertain-label", type=str, default="uncertain")

    parser.add_argument("--n-configs-per-family", type=int, default=4)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--image-selection-mode", choices=["best", "worst"], default="best")
    parser.add_argument("--image-selection-from", choices=["2way", "3way", "auto"], default="2way")
    parser.add_argument("--max-single-target-images", type=int, default=1)
    parser.add_argument("--max-pixels-qt2", type=int, default=8000)

    parser.add_argument("--save-html", action="store_true", default=True)
    parser.add_argument("--no-html", dest="save_html", action="store_false")
    parser.add_argument("--save-png", action="store_true", default=False)
    parser.add_argument("--png-width", type=int, default=1400)
    parser.add_argument("--png-height", type=int, default=1000)
    parser.add_argument("--png-scale", type=int, default=2)
    parser.add_argument("--spatial-plot-height", type=int, default=850)
    parser.add_argument("--stretch-spatial", action="store_true", help="Disable equal aspect ratio for spatial maps.")

    parser.add_argument("--only", choices=["all", "2way", "3way"], default="all")
    args = parser.parse_args()

    project_root = find_project_root(args.project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Project imports. These intentionally use the existing project modules.
    from src.io.database_h5 import load_nir_uco_h5
    from src.decision.labels import predicted_col
    from src.visualization.plot_decision import (
        plot_object_decision_map,
        plot_object_error_overlay,
        plot_pixel_prediction_overlay,
        plot_pixel_error_overlay,
        plot_pixel_three_way_decision_overlay,
        plot_confusion_heatmap_from_long,
    )
    from src.visualization.plot_simca import plot_simca_q_t2_dataframe

    target_class = str(args.target_class)
    non_target_label = str(args.non_target_label)
    uncertain_label = str(args.uncertain_label)

    results_dir = args.results_dir or (project_root / "results" / f"05_simca_mixture_application_{args.results_tag}")
    db_h5 = args.db_h5 or (project_root / "HSI Data" / "processed" / "nir_uco_database.h5")
    output_root = args.output_root or (results_dir / "figures")
    output_root.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT:", project_root)
    print("RESULTS_DIR:", results_dir)
    print("DB_H5:", db_h5)
    print("OUTPUT_ROOT:", output_root)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not db_h5.exists():
        raise FileNotFoundError(f"Database not found: {db_h5}")

    # Load image/object data for spatial overlays.
    object_db, image_db = load_nir_uco_h5(
        db_h5,
        reconstruct_heavy_object_arrays=True,
    )

    # Saved projection/metric tables.
    reference_df, reference_path = first_existing_parquet(
        [
            results_dir / "mixture_model_summary.parquet",
            results_dir / "frozen_reference_configs.parquet",
        ],
        required=False,
    )

    object_df = read_parquet_if_exists(results_dir / "mixture_object_predictions.parquet")
    object_3way_df = read_parquet_if_exists(results_dir / "mixture_object_predictions_3way.parquet")

    pixel_df, pixel_path = first_existing_parquet(
        [
            results_dir / "mixture_pixel_predictions_3way_from_object_decision.parquet",
            results_dir / "mixture_pixel_predictions_minimal.parquet",
            results_dir / "mixture_pixel_predictions.parquet",
        ],
        required=False,
    )

    object_2way_confusion_df = read_parquet_if_exists(results_dir / "mixture_object_2way_confusion.parquet")
    pixel_2way_confusion_df = read_parquet_if_exists(results_dir / "mixture_pixel_2way_confusion.parquet")
    object_3way_confusion_df = read_parquet_if_exists(results_dir / "mixture_object_3way_confusion.parquet")
    pixel_3way_confusion_df = read_parquet_if_exists(results_dir / "mixture_pixel_3way_confusion_from_object_decision.parquet")

    object_errors_by_image_df = read_parquet_if_exists(results_dir / "mixture_object_errors_by_image.parquet")
    pixel_errors_by_image_df = read_parquet_if_exists(results_dir / "mixture_pixel_errors_by_image.parquet")
    three_way_by_image_df = read_parquet_if_exists(results_dir / "mixture_three_way_by_image.parquet")

    print("Reference table:", reference_path, reference_df.shape)
    print("Object predictions:", object_df.shape)
    print("Object 3-way predictions:", object_3way_df.shape)
    print("Pixel predictions:", pixel_path, pixel_df.shape)

    object_plot_df = object_3way_df if len(object_3way_df) > 0 else object_df
    if len(object_plot_df) == 0:
        raise FileNotFoundError("No object prediction table found. Run notebook 05 first.")

    if len(reference_df) == 0:
        reference_cols = [
            c for c in [
                "selected_config_id",
                "matrix_family",
                "candidate_source",
                "frozen_reference_rank",
                "training_matrix_id",
                "matrix_method",
                "preprocessing",
                "selected_rule_name",
            ]
            if c in object_plot_df.columns
        ]
        reference_df = object_plot_df[reference_cols].drop_duplicates("selected_config_id").copy()
        reference_df["frozen_reference_rank"] = np.arange(1, len(reference_df) + 1)

    require_columns(reference_df, ["selected_config_id"], "reference_df")
    diagnostic_configs_df = choose_diagnostic_configs(
        reference_df,
        n_per_family=args.n_configs_per_family,
        max_configs=args.max_configs,
    )

    if len(diagnostic_configs_df) == 0:
        raise RuntimeError("No diagnostic configuration found.")

    print("Diagnostic configs:", diagnostic_configs_df.shape)

    # Choose by-image source for image ranking.
    if args.image_selection_from == "3way":
        by_image_for_selection = three_way_by_image_df
        metric_preferences = [
            ("target_miss_rate", "low"),
            ("non_target_false_accept_rate", "low"),
            ("uncertain_rate", "low"),
            ("coverage_rate", "high"),
        ]
    elif args.image_selection_from == "auto" and len(object_errors_by_image_df) == 0 and len(three_way_by_image_df) > 0:
        by_image_for_selection = three_way_by_image_df
        metric_preferences = [
            ("target_miss_rate", "low"),
            ("non_target_false_accept_rate", "low"),
            ("uncertain_rate", "low"),
            ("coverage_rate", "high"),
        ]
    else:
        by_image_for_selection = object_errors_by_image_df
        metric_preferences = [
            ("fn_rate", "low"),
            ("fp_rate", "low"),
            ("balanced_accuracy", "high"),
        ]

    pred_object_col = predicted_col(target_class, "object")
    pred_pixel_col = predicted_col(target_class, "pixel")

    TWO_WAY_ORDER = [non_target_label, target_class]
    TWO_WAY_COLORS = {
        non_target_label: "royalblue",
        target_class: "limegreen",
    }
    THREE_WAY_ORDER = [non_target_label, uncertain_label, target_class]
    THREE_WAY_COLORS = {
        non_target_label: "royalblue",
        uncertain_label: "purple",
        target_class: "limegreen",
    }

    summary_rows = []

    for _, cfg in diagnostic_configs_df.iterrows():
        config_id = str(cfg["selected_config_id"])
        config_dir = output_root / make_config_folder_name(cfg)
        config_dir.mkdir(parents=True, exist_ok=True)

        cfg.to_frame().T.to_csv(config_dir / "config_info.csv", index=False)

        obj_config = object_plot_df[object_plot_df["selected_config_id"].astype(str).eq(config_id)].copy()
        obj_binary = add_plot_label_columns(
            obj_config,
            target_class=target_class,
            non_target_label=non_target_label,
            uncertain_label=uncertain_label,
        )
        obj_3way = (
            object_3way_df[
                object_3way_df["selected_config_id"].astype(str).eq(config_id)
            ].copy()
            if len(object_3way_df) > 0
            else pd.DataFrame()
        )
        obj_3way = add_plot_label_columns(
            obj_3way,
            target_class=target_class,
            non_target_label=non_target_label,
            uncertain_label=uncertain_label,
        )
        pix_config = (
            pixel_df[
                pixel_df["selected_config_id"].astype(str).eq(config_id)
            ].copy()
            if len(pixel_df) > 0 and "selected_config_id" in pixel_df.columns
            else pd.DataFrame()
        )
        pix_config = add_plot_label_columns(
            pix_config,
            target_class=target_class,
            non_target_label=non_target_label,
            uncertain_label=uncertain_label,
        )
        selected_images_df = choose_images_for_config(
            config_id=config_id,
            by_image_df=by_image_for_selection,
            object_df_config=obj_config,
            metric_preferences=metric_preferences,
            n_images=args.n_images,
            mode=args.image_selection_mode,
            max_single_target_images=args.max_single_target_images,
            target_class=target_class,
        )
        selected_images_df.to_csv(config_dir / "selected_images.csv", index=False)
        image_keys = selected_images_df["source_image"].astype(str).tolist() if len(selected_images_df) > 0 else []

        print("\n" + "=" * 100)
        print(config_id)
        print("Output:", config_dir)
        print("Selected images:", image_keys)

        summary_rows.append({
            "selected_config_id": config_id,
            "config_dir": str(config_dir),
            "n_selected_images": len(image_keys),
            "selected_images": ";".join(image_keys),
        })

        # ------------------------------------------------------------------
        # 2-way figures
        # ------------------------------------------------------------------
        if args.only in {"all", "2way"}:
            fig = plot_saved_confusion(
                object_2way_confusion_df,
                config_id=config_id,
                mode="2way",
                title=f"Object-level 2-way confusion — {config_id}",
                plot_confusion_heatmap_from_long=plot_confusion_heatmap_from_long,
            )
            save_plotly_figure(fig, "00_object_2way_confusion", config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale)

            fig = plot_saved_confusion(
                pixel_2way_confusion_df,
                config_id=config_id,
                mode="2way",
                title=f"Pixel-level 2-way confusion — {config_id}",
                plot_confusion_heatmap_from_long=plot_confusion_heatmap_from_long,
            )
            save_plotly_figure(fig, "01_pixel_2way_confusion", config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale)

            for image_key in image_keys:
                try:
                    fig = plot_object_decision_map(
                        image_db=image_db,
                        object_db=object_db,
                        results_df=obj_binary,
                        image_key=image_key,
                        decision_col="predicted_label_object_plot" if "predicted_label_object_plot" in obj_binary.columns else "predicted_label_object",
                        decision_to_code={non_target_label: 1, target_class: 2},
                        code_to_name={1: non_target_label, 2: target_class},
                        title=f"Objectwise 2-way decision — {config_id} — {image_key}",
                        show=False,
                    )
                    save_spatial_figure(fig, f"2way_{image_key}_object_decision_map", image_db, image_key, config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale, args.spatial_plot_height, not args.stretch_spatial)
                except Exception as exc:
                    print(f"  [WARNING] Object 2-way decision map failed for {config_id}/{image_key}: {exc!r}")

                try:
                    fig = plot_object_error_overlay(
                        image_key=image_key,
                        image_db=image_db,
                        object_db=object_db,
                        object_df=obj_binary,
                        target_class=target_class,
                        title=f"Object-level 2-way errors — {config_id} — {image_key}",
                        show=False,
                    )
                    save_spatial_figure(fig, f"2way_{image_key}_object_errors", image_db, image_key, config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale, args.spatial_plot_height, not args.stretch_spatial)
                except Exception as exc:
                    print(f"  [WARNING] Object 2-way errors failed for {config_id}/{image_key}: {exc!r}")

                if len(pix_config) > 0:
                    try:
                        fig = plot_pixel_prediction_overlay(
                            image_key=image_key,
                            image_db=image_db,
                            pixel_df=pix_config,
                            target_class=target_class,
                            pred_col=pred_pixel_col if pred_pixel_col in pix_config.columns else None,
                            title=f"Pixel-level 2-way prediction — {config_id} — {image_key}",
                            show=False,
                        )
                        save_spatial_figure(fig, f"2way_{image_key}_pixel_prediction_map", image_db, image_key, config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale, args.spatial_plot_height, not args.stretch_spatial)
                    except Exception as exc:
                        print(f"  [WARNING] Pixel 2-way prediction map failed for {config_id}/{image_key}: {exc!r}")

                    try:
                        fig = plot_pixel_error_overlay(
                            image_key=image_key,
                            image_db=image_db,
                            pixel_df=pix_config,
                            target_class=target_class,
                            title=f"Pixel-level 2-way errors — {config_id} — {image_key}",
                            show=False,
                        )
                        save_spatial_figure(fig, f"2way_{image_key}_pixel_errors", image_db, image_key, config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale, args.spatial_plot_height, not args.stretch_spatial)
                    except Exception as exc:
                        print(f"  [WARNING] Pixel 2-way errors failed for {config_id}/{image_key}: {exc!r}")

            try:
                fig = plot_simca_q_t2_dataframe(
                    obj_binary,
                    level="object",
                    label_col="predicted_label_object_plot"
                    if "predicted_label_object_plot" in obj_binary.columns
                    else "predicted_label_object",
                    confidence_col="binary_confidence"
                    if "binary_confidence" in obj_binary.columns
                    else None,
                    title=f"Object-level 2-way SIMCA Q residuals vs Hotelling T² — {config_id}",
                    category_order=TWO_WAY_ORDER,
                    color_map=TWO_WAY_COLORS,
                    force_legend_groups=True,
                    show=False,
                )
                save_plotly_figure(fig, "2way_object_qres_t2", config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale)
            except Exception as exc:
                print(f"  [WARNING] Object 2-way Q/T² failed for {config_id}: {exc!r}")

            if len(pix_config) > 0:
                try:
                    label_col = "predicted_label_pixel" if "predicted_label_pixel" in pix_config.columns else pred_pixel_col
                    pix_plot = sample_for_qt2(pix_config, label_col=label_col, max_rows=args.max_pixels_qt2)
                    fig = plot_simca_q_t2_dataframe(
                        pix_plot,
                        level="pixel",
                        label_col=label_col,
                        confidence_col="binary_confidence" if "binary_confidence" in pix_plot.columns else None,
                        title=f"Pixel-level 2-way SIMCA Q residuals vs Hotelling T² — {config_id}",
                        category_order=TWO_WAY_ORDER,
                        color_map=TWO_WAY_COLORS,
                        force_legend_groups=True,
                        show=False,
                    )
                    save_plotly_figure(fig, "2way_pixel_qres_t2", config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale)
                except Exception as exc:
                    print(f"  [WARNING] Pixel 2-way Q/T² failed for {config_id}: {exc!r}")

        # ------------------------------------------------------------------
        # 3-way figures
        # ------------------------------------------------------------------
        if args.only in {"all", "3way"} and len(obj_3way) > 0:
            fig = plot_saved_confusion(
                object_3way_confusion_df,
                config_id=config_id,
                mode="3way",
                title=f"Object-level 3-way confusion — {config_id}",
                plot_confusion_heatmap_from_long=plot_confusion_heatmap_from_long,
            )
            save_plotly_figure(fig, "00_object_3way_confusion", config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale)

            fig = plot_saved_confusion(
                pixel_3way_confusion_df,
                config_id=config_id,
                mode="3way",
                title=f"Pixel-level 3-way confusion — {config_id}",
                plot_confusion_heatmap_from_long=plot_confusion_heatmap_from_long,
            )
            save_plotly_figure(fig, "01_pixel_3way_confusion", config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale)

            for image_key in image_keys:
                try:
                    fig = plot_object_decision_map(
                        image_db=image_db,
                        object_db=object_db,
                        results_df=obj_3way,
                        image_key=image_key,
                        decision_col="decision_3way_plot" if "decision_3way_plot" in obj_3way.columns else "decision_3way",
                        decision_to_code={
                            non_target_label: 1,
                            uncertain_label: 2,
                            target_class: 3,
                        },
                        code_to_name={
                            1: non_target_label,
                            2: uncertain_label,
                            3: target_class,
                        },
                        title=f"Objectwise 3-way decision — {config_id} — {image_key}",
                        show=False,
                    )
                    save_spatial_figure(fig, f"3way_{image_key}_object_decision_map", image_db, image_key, config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale, args.spatial_plot_height, not args.stretch_spatial)
                except Exception as exc:
                    print(f"  [WARNING] Object 3-way decision map failed for {config_id}/{image_key}: {exc!r}")

                if len(pix_config) > 0 and "decision_3way" in pix_config.columns:
                    try:
                        fig = plot_pixel_three_way_decision_overlay(
                            image_key=image_key,
                            image_db=image_db,
                            pixel_df=pix_config,
                            decision_col="decision_3way_plot" if "decision_3way_plot" in pix_config.columns else "decision_3way",
                            target_class=target_class,
                            non_target_label=non_target_label,
                            uncertain_label=uncertain_label,
                            title=f"Pixel view of objectwise 3-way decision — {config_id} — {image_key}",
                            show=False,
                        )
                        save_spatial_figure(fig, f"3way_{image_key}_pixel_decision_map", image_db, image_key, config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale, args.spatial_plot_height, not args.stretch_spatial)
                    except Exception as exc:
                        print(f"  [WARNING] Pixel 3-way decision map failed for {config_id}/{image_key}: {exc!r}")

            try:
                fig = plot_simca_q_t2_dataframe(
                    obj_3way,
                    level="object",
                    label_col="decision_3way_plot" if "decision_3way_plot" in obj_3way.columns else "decision_3way",
                    confidence_col="three_way_confidence" if "three_way_confidence" in obj_3way.columns else None,
                    title=f"Object-level 3-way SIMCA Q residuals vs Hotelling T² — {config_id}",
                    category_order=THREE_WAY_ORDER,
                    color_map=THREE_WAY_COLORS,
                    force_legend_groups=True,
                    show=False,
                )
                save_plotly_figure(fig, "3way_object_qres_t2", config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale)
            except Exception as exc:
                print(f"  [WARNING] Object 3-way Q/T² failed for {config_id}: {exc!r}")

            if len(pix_config) > 0 and "decision_3way" in pix_config.columns:
                try:
                    pix_plot = sample_for_qt2(pix_config, label_col="decision_3way_plot" if "decision_3way_plot" in pix_config.columns else "decision_3way", max_rows=args.max_pixels_qt2)
                    fig = plot_simca_q_t2_dataframe(
                        pix_plot,
                        level="pixel",
                        label_col="decision_3way_plot" if "decision_3way_plot" in pix_plot.columns else "decision_3way",
                        confidence_col="three_way_confidence" if "three_way_confidence" in pix_plot.columns else None,
                        title=f"Pixel-level 3-way SIMCA Q residuals vs Hotelling T² — {config_id}",
                        category_order=THREE_WAY_ORDER,
                        color_map=THREE_WAY_COLORS,
                        force_legend_groups=True,
                        show=False,
                    )
                    save_plotly_figure(fig, "3way_pixel_qres_t2", config_dir, args.save_html, args.save_png, args.png_width, args.png_height, args.png_scale)
                except Exception as exc:
                    print(f"  [WARNING] Pixel 3-way Q/T² failed for {config_id}: {exc!r}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "figure_generation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print("Figure root:", output_root)
    print("Summary:", summary_path)


# Keep this helper near main to avoid forward references in the script.
def sample_for_qt2(
    df: pd.DataFrame,
    label_col: str,
    max_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    max_rows = int(max_rows)
    if len(df) <= max_rows:
        return df.copy()

    if label_col in df.columns:
        n_groups = max(df[label_col].nunique(dropna=False), 1)
        n_per_group = max(1, max_rows // n_groups)
        return (
            df.groupby(label_col, group_keys=False, dropna=False)
            .apply(lambda g: g.sample(n=min(len(g), n_per_group), random_state=random_state))
            .reset_index(drop=True)
        )

    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


if __name__ == "__main__":
    main()
