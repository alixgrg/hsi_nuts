from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.utils import as_1d_array
from src.visualization.common import (
    make_customdata,
    make_dynamic_color_map,
    show_or_return,
    validate_columns,
)

def plot_scores(
    scores,
    #pca_res=None,
    dims: Sequence[int] | None = None,
    pcx: int = 1,
    pcy: int = 2,
    pcz: int | None = None,
    labels=None,
    color_values=None,
    color_by: str = "label",
    color_sequence=None,
    continuous_colorscale: str = "Turbo",
    object_ids=None,
    source_images=None,
    batches=None,
    areas=None,
    subset = None,
    title: str | None = None,
    width: int = 850,
    height: int = 650,
    show: bool = True,
    symbol_values=None,
    symbol_by: str | None = None,
    contour_values=None,
    contour_by: str | None = None,
    use_open_symbol_for_contour: bool = True,
    marker_size: int = 9,
    marker_opacity: float = 0.85,
    **metadata,
):
    if dims is None:
        dims = (pcx, pcy) if pcz is None else (pcx, pcy, pcz)
    dims = tuple(dims)
    idx = [d - 1 for d in dims]
    n = scores.shape[0]

    if color_values is None:
        if color_by == "source_image":
            color_values = source_images
        elif color_by == "batch":
            color_values = batches
        elif color_by == "subset":
            color_values = subset
        else:
            color_values = labels
    color_groups = as_1d_array(color_values, n, "all").astype(str)

    color_map = make_dynamic_color_map(
        color_groups,
        color_sequence=color_sequence,
        continuous_colorscale=continuous_colorscale,
    )

    if symbol_values is None:
        if symbol_by == "batch":
            symbol_values = batches
        elif symbol_by == "source_image":
            symbol_values = source_images
        elif symbol_by == "subset":
            symbol_values = subset
        elif symbol_by == "label":
            symbol_values = labels
        else:
            symbol_values = np.array(["all"] * n)
    symbol_groups = as_1d_array(symbol_values, n, "all").astype(str)

    if contour_values is None:
        if contour_by == "subset":
            contour_values = subset
        elif contour_by == "batch":
            contour_values = batches
        elif contour_by == "source_image":
            contour_values = source_images
        elif contour_by == "label":
            contour_values = labels
        else:
            contour_values = np.array(["filled"] * n)
    contour_groups = as_1d_array(contour_values, n, "filled").astype(str)

    meta = dict(
        object_id=object_ids,
        label=labels,
        source_image=source_images,
        batch=batches,
        subset=subset,
        area=areas,
    )
    meta.update(metadata)
    custom, hover_meta = make_customdata(n, **meta)

    base_symbols = [
        "circle",
        "square",
        "diamond",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "star",
    ]
    unique_symbol_groups = np.unique(symbol_groups)
    symbol_map = {
        group: base_symbols[i % len(base_symbols)]
        for i, group in enumerate(unique_symbol_groups)
    }

    def is_open_contour(contour_group):
        contour_group_lower = str(contour_group).lower()
        open_keywords = [
            "projection",
            "test",
            "validation",
            "val",
            "external",
            "projected",
        ]
        return any(key in contour_group_lower for key in open_keywords)

    def symbol_with_contour(base_symbol, contour_group):
        if not use_open_symbol_for_contour:
            return base_symbol
        if is_open_contour(contour_group):
            return f"{base_symbol}-open"
        return base_symbol

    def contour_line_width(contour_group):
        return 2.0 if is_open_contour(contour_group) else 0.8

    def contour_line_color(contour_group, color_group):
        # Open markers: outline keeps the class color.
        # Filled markers: black outline improves readability.
        if is_open_contour(contour_group):
            return color_map[color_group]
        return "black"

    fig = go.Figure()
    combined_groups = np.array([
        f"{c}|||{s}|||{k}"
        for c, s, k in zip(color_groups, symbol_groups, contour_groups)
    ])
    unique_combined = np.unique(combined_groups)
    if len(dims) == 2:
        if title is None:
            title = f"Scores: C{dims[0]} vs C{dims[1]}"
        for combined in unique_combined:
            color_group, symbol_group, contour_group = combined.split("|||")
            mask = combined_groups == combined
            base_symbol = symbol_map[symbol_group]
            marker_symbol = symbol_with_contour(base_symbol, contour_group)
            trace_name = f"{color_group} | batch={symbol_group} | set={contour_group}"
            fig.add_trace(
                go.Scatter(
                    x=scores[mask, idx[0]],
                    y=scores[mask, idx[1]],
                    mode="markers",
                    name=trace_name,
                    customdata=custom[mask],
                    marker=dict(
                        size=marker_size,
                        opacity=marker_opacity,
                        symbol=marker_symbol,
                        color=color_map[color_group],
                        line=dict(
                            width=contour_line_width(contour_group),
                            color=contour_line_color(contour_group, color_group),
                        ),
                    ),
                    hovertemplate=(
                        f"C{dims[0]}: %{{x:.4f}}<br>"
                        f"C{dims[1]}: %{{y:.4f}}<br>"
                        + hover_meta
                        + "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title=f"C{dims[0]}",
            yaxis_title=f"C{dims[1]}",
            width=width,
            height=height,
            legend_title_text="Color | Batch | Set",
        )
    elif len(dims) == 3:
        if title is None:
            title = f"Scores: C{dims[0]} vs C{dims[1]} vs C{dims[2]}"
        for combined in unique_combined:
            color_group, symbol_group, contour_group = combined.split("|||")
            mask = combined_groups == combined
            # Scatter3d supports fewer marker symbols than Scatter.
            # We keep the batch symbol, but open-symbol rendering is less reliable in 3D.
            base_symbol = symbol_map[symbol_group]
            marker_symbol = base_symbol
            trace_name = f"{color_group} | batch={symbol_group} | set={contour_group}"
            fig.add_trace(
                go.Scatter3d(
                    x=scores[mask, idx[0]],
                    y=scores[mask, idx[1]],
                    z=scores[mask, idx[2]],
                    mode="markers",
                    name=trace_name,
                    customdata=custom[mask],
                    marker=dict(
                        size=max(marker_size - 3, 4),
                        opacity=marker_opacity,
                        symbol=marker_symbol,
                        color=color_map[color_group],
                        line=dict(
                            width=contour_line_width(contour_group),
                            color=contour_line_color(contour_group, color_group),
                        ),
                    ),
                    hovertemplate=(
                        f"C{dims[0]}: %{{x:.4f}}<br>"
                        f"C{dims[1]}: %{{y:.4f}}<br>"
                        f"C{dims[2]}: %{{z:.4f}}<br>"
                        + hover_meta
                        + "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            legend_title_text="Color | Batch | Set",
            scene=dict(
                xaxis_title=f"C{dims[0]}",
                yaxis_title=f"C{dims[1]}",
                zaxis_title=f"C{dims[2]}",
            ),
        )
    else:
        raise ValueError("dims must contain 2 or 3 components.")
    return show_or_return(fig, show)


# SCORES FOR PIXEL-LEVEL PCA

def build_scores_dataframe(
    scores,
    labels=None,
    meta=None,
    subset=None,
    dims=(1, 2, 3),
    score_prefix="C",
):
    """
    Build a tidy DataFrame from PCA / SIMCA / latent scores.

    Parameters
    ----------
    scores : array-like, shape (n_samples, n_components)
        Score matrix.
    labels : array-like, optional
        Class labels, e.g. nut type.
    meta : DataFrame or dict, optional
        Metadata associated with each observation.
    subset : array-like, optional
        Set information, e.g. train / projection / test.
    dims : tuple
        Components to include. Components are 1-indexed.
    score_prefix : str
        Prefix for component columns, e.g. "C" or "PC".

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with score columns and metadata.
    """
    scores = np.asarray(scores)
    n = scores.shape[0]
    if meta is None:
        df = pd.DataFrame(index=np.arange(n))
    elif isinstance(meta, dict):
        df = pd.DataFrame(meta).copy()
    else:
        df = meta.copy()
    if len(df) != n:
        raise ValueError(
            f"Metadata length ({len(df)}) does not match scores length ({n})."
        )
    for d in dims:
        idx = d - 1
        if idx >= scores.shape[1]:
            raise ValueError(
                f"Component {d} requested, but scores only have "
                f"{scores.shape[1]} components."
            )
        df[f"{score_prefix}{d}"] = scores[:, idx]
    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != n:
            raise ValueError(
                f"Labels length ({len(labels)}) does not match scores length ({n})."
            )
        df["label"] = labels.astype(str)
    if subset is not None:
        subset = np.asarray(subset)
        if len(subset) != n:
            raise ValueError(
                f"Subset length ({len(subset)}) does not match scores length ({n})."
            )
        df["subset"] = subset.astype(str)
    # Convert useful metadata to string for robust plotting
    for col in ["object_id", "source_image", "batch", "label", "subset"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def sample_scores_dataframe(
    df,
    group_cols=None,
    n_per_group=500,
    random_state=0,
    keep_group_cols=True,
):
    """
    Balanced sampling of a score DataFrame.

    This version preserves all original columns, including grouping columns
    such as label, subset and batch.
    """
    if group_cols is None:
        group_cols = []

    df = df.copy()
    # Keep only existing columns
    group_cols = [col for col in group_cols if col in df.columns]
    # If no valid grouping columns, simple random sample
    if len(group_cols) == 0:
        return (
            df.sample(
                n=min(len(df), n_per_group),
                random_state=random_state,
            )
            .reset_index(drop=True)
        )

    sampled_parts = []
    for _, group in df.groupby(group_cols, dropna=False, sort=False):
        sampled_group = group.sample(
            n=min(len(group), n_per_group),
            random_state=random_state,
        )
        sampled_parts.append(sampled_group)
    df_sample = pd.concat(sampled_parts, axis=0, ignore_index=True)
    # Safety check: grouping columns should still be present
    if keep_group_cols:
        missing_cols = [col for col in group_cols if col not in df_sample.columns]
        if missing_cols:
            raise RuntimeError(
                f"Sampling removed grouping columns: {missing_cols}. "
                "Check the input DataFrame and groupby logic."
            )
    return df_sample

def plot_scores_density(
    df,
    x="C1",
    y="C2",
    color_by="label",
    facet_col=None,
    facet_row=None,
    mode="contour",
    nbinsx=80,
    nbinsy=80,
    title=None,
    width=950,
    height=650,
    show=True,
):
    """
    Plot 2D density of scores.

    Parameters
    ----------
    df : pandas.DataFrame
        Score DataFrame.
    x, y : str
        Score columns.
    color_by : str
        Column used for contours color.
    facet_col, facet_row : str, optional
        Columns used for faceting.
    mode : {"contour", "heatmap"}
        Density visualization mode.
    nbinsx, nbinsy : int
        Number of bins for heatmap mode.
    """
    if title is None:
        title = f"Score density: {x} vs {y}"
    if mode == "contour":
        fig = px.density_contour(
            df,
            x=x,
            y=y,
            color=color_by if color_by in df.columns else None,
            facet_col=facet_col if facet_col in df.columns else None,
            facet_row=facet_row if facet_row in df.columns else None,
            title=title,
        )
        fig.update_traces(contours_coloring="none")
    elif mode == "heatmap":
        fig = px.density_heatmap(
            df,
            x=x,
            y=y,
            facet_col=facet_col if facet_col in df.columns else None,
            facet_row=facet_row if facet_row in df.columns else None,
            nbinsx=nbinsx,
            nbinsy=nbinsy,
            title=title,
        )
    else:
        raise ValueError("mode must be either 'contour' or 'heatmap'.")
    fig.update_layout(width=width, height=height)
    if show:
        fig.show()
    return fig


def plot_scores_distribution(
    df,
    score_col="C1",
    x_by="label",
    color_by="label",
    facet_col=None,
    facet_row=None,
    kind="violin",
    box=True,
    points=False,
    title=None,
    width=950,
    height=600,
    show=True,
):
    """
    Plot score distributions by class, batch, subset, etc.

    Parameters
    ----------
    df : pandas.DataFrame
        Score DataFrame.
    score_col : str
        Score column to plot.
    x_by : str
        Grouping column on x-axis.
    color_by : str
        Color grouping column.
    facet_col, facet_row : str, optional
        Faceting columns.
    kind : {"violin", "box", "histogram"}
        Type of distribution plot.
    """
    if title is None:
        title = f"Distribution of {score_col}"
    facet_col_arg = facet_col if facet_col in df.columns else None
    facet_row_arg = facet_row if facet_row in df.columns else None
    color_arg = color_by if color_by in df.columns else None
    if kind == "violin":
        fig = px.violin(
            df,
            x=x_by,
            y=score_col,
            color=color_arg,
            facet_col=facet_col_arg,
            facet_row=facet_row_arg,
            box=box,
            points="all" if points else False,
            title=title,
        )
    elif kind == "box":
        fig = px.box(
            df,
            x=x_by,
            y=score_col,
            color=color_arg,
            facet_col=facet_col_arg,
            facet_row=facet_row_arg,
            points="all" if points else False,
            title=title,
        )
    elif kind == "histogram":
        fig = px.histogram(
            df,
            x=score_col,
            color=color_arg,
            facet_col=facet_col_arg,
            facet_row=facet_row_arg,
            marginal="box",
            histnorm="probability density",
            opacity=0.65,
            title=title,
        )
    else:
        raise ValueError("kind must be 'violin', 'box', or 'histogram'.")
    fig.update_layout(width=width, height=height)
    if show:
        fig.show()
    return fig


def summarize_scores_by_object(
    df,
    score_cols=("C1", "C2", "C3"),
    object_col="object_id",
    extra_group_cols=None,
):
    """
    Aggregate pixel-level scores into object-level summaries.

    Parameters
    ----------
    df : pandas.DataFrame
        Pixel score DataFrame.
    score_cols : tuple[str]
        Score columns to summarize.
    object_col : str
        Object identifier column.
    extra_group_cols : list[str], optional
        Additional columns to preserve, e.g. label, subset, batch, source_image.

    Returns
    -------
    summary : pandas.DataFrame
        One row per object.
    """
    if object_col not in df.columns:
        raise ValueError(f"Column '{object_col}' not found in DataFrame.")
    if extra_group_cols is None:
        extra_group_cols = ["label", "subset", "batch", "source_image"]
    group_cols = [object_col] + [
        col for col in extra_group_cols
        if col in df.columns and col != object_col
    ]
    score_cols = [col for col in score_cols if col in df.columns]
    agg_dict = {}
    for col in score_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")
        agg_dict[f"{col}_median"] = (col, "median")
        agg_dict[f"{col}_q05"] = (col, lambda s: s.quantile(0.05))
        agg_dict[f"{col}_q95"] = (col, lambda s: s.quantile(0.95))
    # Pixel count
    first_score = score_cols[0]
    agg_dict["n_pixels"] = (first_score, "size")
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(**agg_dict)
    )
    return summary


def plot_object_score_summary(
    df_object,
    x="C1_mean",
    y="C2_mean",
    color_by="label",
    symbol_by="batch",
    facet_col="subset",
    size_by="n_pixels",
    error_x=None,
    error_y=None,
    hover_cols=None,
    title=None,
    width=950,
    height=650,
    show=True,
):
    """
    Plot object-level summary of pixel scores.

    Parameters
    ----------
    df_object : pandas.DataFrame
        Output of summarize_scores_by_object().
    x, y : str
        Columns used as axes.
    color_by : str
        Color column.
    symbol_by : str
        Symbol column.
    facet_col : str, optional
        Facet column.
    size_by : str, optional
        Marker size column.
    error_x, error_y : str, optional
        Error bar columns.
    """
    if title is None:
        title = f"Object summary of pixel scores: {x} vs {y}"
    if hover_cols is None:
        hover_cols = [
            col for col in [
                "object_id",
                "source_image",
                "batch",
                "subset",
                "label",
                "n_pixels",
                "C1_std",
                "C2_std",
            ]
            if col in df_object.columns
        ]
    fig = px.scatter(
        df_object,
        x=x,
        y=y,
        color=color_by if color_by in df_object.columns else None,
        symbol=symbol_by if symbol_by in df_object.columns else None,
        facet_col=facet_col if facet_col in df_object.columns else None,
        size=size_by if size_by in df_object.columns else None,
        error_x=error_x if error_x in df_object.columns else None,
        error_y=error_y if error_y in df_object.columns else None,
        hover_data=hover_cols,
        title=title,
    )
    fig.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(width=1, color="black"),
        )
    )
    fig.update_layout(width=width, height=height)
    if show:
        fig.show()
    return fig
