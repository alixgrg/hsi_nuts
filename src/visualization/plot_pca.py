from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots

from src.utils import wavelength_axis
from src.visualization.common import (
    apply_project_theme,
    make_dynamic_color_map,
    ordered_unique,
    show_or_return,
)
from src.visualization.plot_diagnostics import (
    plot_metric_by_index,
    plot_metric_heatmap,
    plot_xy_diagnostic,
)
from src.visualization.plot_scores import plot_scores


def plot_pca_review_panel(
    result: Mapping,
    *,
    wavelengths=None,
    figure=None,
    axes=None,
):
    """Render the six-panel visual evidence required for one PCA candidate."""
    if figure is None or axes is None:
        figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = np.asarray(axes).reshape(2, 3)
    X = np.asarray(result["X_preprocessed"], dtype=float)
    scores = np.asarray(result["scores"], dtype=float)
    loadings = np.asarray(result["loadings"], dtype=float)
    labels = np.asarray(result["y"]).astype(str)
    axis = np.arange(X.shape[1]) if wavelengths is None else np.asarray(wavelengths)

    spectra_ax = axes[0, 0]
    noise_ax = axes[0, 1]
    scores_ax = axes[0, 2]
    loadings_ax = axes[1, 0]
    variance_ax = axes[1, 1]
    distance_ax = axes[1, 2]
    for label in np.unique(labels):
        values = X[labels == label]
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        spectra_ax.plot(axis, mean, label=label)
        spectra_ax.fill_between(axis, mean - std, mean + std, alpha=0.15)
        noise_ax.plot(axis, std, label=label)
        scores_ax.scatter(
            scores[labels == label, 0],
            scores[labels == label, min(1, scores.shape[1] - 1)],
            s=10,
            alpha=0.45,
            label=label,
        )
    spectra_ax.set_title("Spectres prétraités (moyenne ± écart-type)")
    noise_ax.set_title("Bruit / dispersion spectrale")
    scores_ax.set_title("Scores PC1–PC2")
    spectra_ax.legend(fontsize=8)
    scores_ax.legend(fontsize=8)

    for component in range(min(3, loadings.shape[1])):
        loadings_ax.plot(axis, loadings[:, component], label=f"PC{component + 1}")
    loadings_ax.set_title("Loadings")
    loadings_ax.legend(fontsize=8)

    evr = np.asarray(result["explained_variance_ratio"], dtype=float)
    shown = min(20, len(evr))
    components = np.arange(1, shown + 1)
    variance_ax.bar(components, evr[:shown], alpha=0.65, label="individuelle")
    variance_ax.plot(components, np.cumsum(evr)[:shown], color="black", label="cumulée")
    variance_ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
    variance_ax.set_title("Variance expliquée")
    variance_ax.legend(fontsize=8)

    pca = result["pca"]
    n_distance_components = min(3, loadings.shape[1])
    t2 = pca.hotelling_t2(X, n_components=n_distance_components)
    q, _ = pca.q_residuals(X, n_components=n_distance_components)
    distance_ax.scatter(t2, q, s=10, alpha=0.45)
    distance_ax.set_xlabel("Hotelling T²")
    distance_ax.set_ylabel("Q résiduel")
    distance_ax.set_title("Distances Q–T²")

    candidate = str(result.get("candidate_id", "candidate"))
    title = (
        f"{candidate} | {result.get('matrix_variant', result.get('matrix_method', ''))}"
        f" | {result.get('preprocessing', '')}"
    )
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure, axes


def build_pca_visual_review_pdf(
    candidate_results: Mapping[str, Mapping],
    candidate_plan: pd.DataFrame,
    output_path: str | Path,
    *,
    wavelengths=None,
) -> dict[str, int]:
    """Write exactly one review page for every technically valid candidate."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_ids = candidate_plan["candidate_id"].astype(str).tolist()
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidate_results]
    if missing:
        raise RuntimeError(
            "The visual review PDF cannot omit technically valid candidates: "
            f"{missing[:10]}"
        )
    page_by_candidate = {}
    with PdfPages(
        output_path,
        metadata={"CreationDate": None, "ModDate": None},
    ) as pdf:
        for page, candidate_id in enumerate(candidate_ids, start=1):
            figure, _ = plot_pca_review_panel(
                candidate_results[candidate_id],
                wavelengths=wavelengths,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            page_by_candidate[candidate_id] = page
    return page_by_candidate


def plot_explained_variance(
    explained_variance_ratio,
    cumulative_explained_variance_ratio=None,
    n_components_to_show: int | None = None,
    thresholds: Sequence[float] = (0.90, 0.95),
    title: str = "Explained variance",
    width: int = 850,
    height: int = 500,
    show: bool = True,
):
    evr = np.asarray(explained_variance_ratio, dtype=float)
    cum = (
        np.cumsum(evr)
        if cumulative_explained_variance_ratio is None
        else np.asarray(cumulative_explained_variance_ratio, dtype=float)
    )
    if n_components_to_show is not None:
        evr = evr[: int(n_components_to_show)]
        cum = cum[: int(n_components_to_show)]
    pcs = np.arange(1, len(evr) + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pcs,
            y=evr,
            name="Explained variance",
            hovertemplate="PC%{x}<br>variance: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pcs,
            y=cum,
            mode="lines+markers",
            name="Cumulative variance",
            hovertemplate="PC%{x}<br>cumulative: %{y:.2%}<extra></extra>",
        )
    )
    for threshold in thresholds:
        fig.add_hline(
            y=float(threshold),
            line_dash="dot",
            annotation_text=f"{threshold:.0%}",
            annotation_position="top left",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Variance ratio",
        width=width,
        height=height,
        barmode="overlay",
    )
    fig.update_yaxes(tickformat=".0%")
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_loadings(
    loadings: np.ndarray,
    wavelengths=None,
    components: Sequence[int] = (1, 2, 3),
    component_names: Sequence[str] | None = None,
    explained_variance_ratio=None,
    title: str = "Loadings",
    width: int = 900,
    height: int = 500,
    show: bool = True,
):
    loadings = np.asarray(loadings, dtype=float)
    if loadings.ndim != 2:
        raise ValueError("loadings must be a 2D array.")
    x, x_title = wavelength_axis(loadings.shape[0], wavelengths)
    evr = None if explained_variance_ratio is None else np.asarray(explained_variance_ratio)

    fig = go.Figure()
    for index, component in enumerate(components):
        column = int(component) - 1
        if column < 0 or column >= loadings.shape[1]:
            continue
        if component_names is not None and index < len(component_names):
            name = component_names[index]
        else:
            name = f"PC{component}"
            if evr is not None and column < len(evr):
                name += f" ({evr[column]:.1%})"
        fig.add_trace(
            go.Scatter(x=x, y=loadings[:, column], mode="lines", name=name)
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Loading",
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_biplot(
    scores: np.ndarray,
    loadings: np.ndarray,
    dims: Sequence[int] = (1, 2),
    labels=None,
    color_by: str = "label",
    color_values=None,
    object_ids=None,
    source_images=None,
    wavelengths=None,
    n_loadings: int = 10,
    loading_scale: float = 1.0,
    explained_variance_ratio=None,
    title: str | None = None,
    show: bool = True,
    **metadata,
):
    scores = np.asarray(scores, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    if len(dims) != 2:
        raise ValueError("Biplot is implemented only in 2D.")
    fig = plot_scores(
        scores,
        dims=dims,
        labels=labels,
        color_values=color_values,
        color_by=color_by,
        object_ids=object_ids,
        source_images=source_images,
        component_variance=explained_variance_ratio,
        title=title or f"Biplot: C{dims[0]} vs C{dims[1]}",
        show=False,
        **metadata,
    )
    ix, iy = int(dims[0]) - 1, int(dims[1]) - 1
    strength = np.sqrt(loadings[:, ix] ** 2 + loadings[:, iy] ** 2)
    top_idx = np.argsort(strength)[-int(n_loadings) :]
    score_range = max(
        np.nanmax(np.abs(scores[:, ix])),
        np.nanmax(np.abs(scores[:, iy])),
    )
    for loading_index in top_idx:
        x_end = loadings[loading_index, ix] * score_range * loading_scale
        y_end = loadings[loading_index, iy] * score_range * loading_scale
        label = (
            f"band {loading_index}"
            if wavelengths is None
            else f"{np.asarray(wavelengths)[loading_index]:.1f} nm"
        )
        fig.add_annotation(
            x=x_end,
            y=y_end,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            text=label,
        )
    return show_or_return(fig, show)


def _require_X(X, function_name: str):
    if X is None:
        raise ValueError(
            f"{function_name} requires the original X matrix. Reconstructing X "
            "from PCA scores would make Q residuals artificially small."
        )
    return np.asarray(X, dtype=float)


def plot_pca_metric_t2(
    pca_model,
    X=None,
    labels=None,
    object_ids=None,
    source_images=None,
    n_components=None,
    title="Hotelling T²",
    show=True,
):
    X = _require_X(X, "plot_pca_metric_t2")
    T2 = pca_model.hotelling_t2(X, n_components=n_components)
    return plot_metric_by_index(
        T2,
        labels=labels,
        object_ids=object_ids,
        source_images=source_images,
        title=title,
        y_title="Hotelling T²",
        show=show,
    )


def plot_pca_metric_q(
    pca_model,
    X=None,
    labels=None,
    object_ids=None,
    source_images=None,
    n_components=None,
    title="Q residuals",
    show=True,
):
    X = _require_X(X, "plot_pca_metric_q")
    Q, _ = pca_model.q_residuals(X, n_components=n_components)
    return plot_metric_by_index(
        Q,
        labels=labels,
        object_ids=object_ids,
        source_images=source_images,
        title=title,
        y_title="Q residual",
        show=show,
    )


def plot_pca_diagnostic(
    pca_model,
    X=None,
    labels=None,
    object_ids=None,
    source_images=None,
    n_components=None,
    title="PCA diagnostic: Q residuals vs Hotelling T²",
    show=True,
):
    X = _require_X(X, "plot_pca_diagnostic")
    T2, Q = pca_model.distances(X, n_components=n_components)
    return plot_xy_diagnostic(
        T2,
        Q,
        labels=labels,
        object_ids=object_ids,
        source_images=source_images,
        title=title,
        x_title="Hotelling T²",
        y_title="Q residual",
        show=show,
    )


def plot_pca_metric_heatmap(
    summary_df,
    metric,
    index_col="preprocessing",
    column_col="matrix_method",
    title=None,
    colorscale="Viridis",
    width=850,
    height=600,
    show=True,
    **kwargs,
):
    if metric not in summary_df.columns:
        raise ValueError(f"Metric {metric!r} not found in summary_df.")
    return plot_metric_heatmap(
        summary_df,
        index_col=index_col,
        columns_col=column_col,
        value_col=metric,
        aggfunc="mean",
        title=title or f"PCA metric heatmap — {metric}",
        colorscale=colorscale,
        width=width,
        height=height,
        show=show,
        **kwargs,
    )


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pareto front for minimising x and maximising y."""
    valid = np.isfinite(x) & np.isfinite(y)
    out = np.zeros(len(x), dtype=bool)
    indices = np.flatnonzero(valid)
    for i in indices:
        dominated = np.any(
            (x[indices] <= x[i])
            & (y[indices] >= y[i])
            & ((x[indices] < x[i]) | (y[indices] > y[i]))
        )
        out[i] = not dominated
    return out


def plot_pca_metric_tradeoff(
    summary_df,
    x_metric="batch_trace_ratio",
    y_metric="class_trace_ratio",
    color_by="matrix_method",
    symbol_by=None,
    size_by=None,
    hover_cols=None,
    highlight_col=None,
    highlight_values=None,
    label_col="preprocessing",
    label_top_n: int = 0,
    show_pareto: bool = False,
    title=None,
    width=900,
    height=650,
    show=True,
):
    """Plot class separation against batch effect with optional selection labels."""
    for metric in (x_metric, y_metric):
        if metric not in summary_df.columns:
            raise ValueError(f"Metric {metric!r} not found.")

    df = summary_df.copy().reset_index(drop=True)
    if hover_cols is None:
        hover_cols = [
            column
            for column in (
                "matrix_method",
                "preprocessing",
                "class_trace_ratio",
                "batch_trace_ratio",
                "class_over_batch_ratio",
                "ncomp_90",
                "ncomp_95",
                "train_q_mean",
                "projection_q_mean",
                "projection_train_q_ratio",
            )
            if column in df.columns
        ]

    color_groups = (
        df[color_by].astype(str)
        if color_by in df.columns
        else pd.Series(["all"] * len(df))
    )
    color_map = make_dynamic_color_map(color_groups)
    symbols = ["circle", "square", "diamond", "triangle-up", "triangle-down"]
    if symbol_by is not None and symbol_by in df.columns:
        symbol_groups = ordered_unique(df[symbol_by].astype(str))
    else:
        symbol_groups = ["all"]
    symbol_map = {value: symbols[index % len(symbols)] for index, value in enumerate(symbol_groups)}

    custom = df[hover_cols].astype(str).to_numpy() if hover_cols else None
    hover = "".join(
        f"{column}: %{{customdata[{index}]}}<br>"
        for index, column in enumerate(hover_cols)
    )

    fig = go.Figure()
    for color_group in ordered_unique(color_groups):
        for symbol_group in symbol_groups:
            mask = color_groups.eq(color_group).to_numpy()
            if symbol_by is not None and symbol_by in df.columns:
                mask &= df[symbol_by].astype(str).eq(symbol_group).to_numpy()
            if not mask.any():
                continue
            marker_size = 10
            if size_by is not None and size_by in df.columns:
                raw = pd.to_numeric(df.loc[mask, size_by], errors="coerce").fillna(0)
                sizes = 8 + 16 * (raw - raw.min()) / max(raw.max() - raw.min(), 1e-12)
            else:
                sizes = marker_size
            fig.add_trace(
                go.Scatter(
                    x=pd.to_numeric(df.loc[mask, x_metric], errors="coerce"),
                    y=pd.to_numeric(df.loc[mask, y_metric], errors="coerce"),
                    mode="markers",
                    name=(
                        color_group
                        if symbol_by is None
                        else f"{color_group} | {symbol_by}={symbol_group}"
                    ),
                    customdata=custom[mask] if custom is not None else None,
                    marker=dict(
                        color=color_map[color_group],
                        symbol=symbol_map[symbol_group],
                        size=sizes,
                        opacity=0.82,
                        line=dict(width=1, color="black"),
                    ),
                    hovertemplate=(
                        f"{x_metric}: %{{x:.4f}}<br>"
                        f"{y_metric}: %{{y:.4f}}<br>"
                        + hover
                        + "<extra></extra>"
                    ),
                )
            )

    highlight_mask = np.zeros(len(df), dtype=bool)
    if highlight_col is not None and highlight_col in df.columns:
        if highlight_values is None:
            highlight_mask = df[highlight_col].fillna(False).astype(bool).to_numpy()
        else:
            values = (
                list(highlight_values)
                if isinstance(highlight_values, (list, tuple, set, np.ndarray))
                else [highlight_values]
            )
            highlight_mask = df[highlight_col].isin(values).to_numpy()
    if highlight_mask.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[highlight_mask, x_metric],
                y=df.loc[highlight_mask, y_metric],
                mode="markers",
                name="selected",
                marker=dict(
                    symbol="circle-open",
                    size=18,
                    color="black",
                    line=dict(width=3, color="black"),
                ),
                hoverinfo="skip",
            )
        )

    if show_pareto:
        x = pd.to_numeric(df[x_metric], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_metric], errors="coerce").to_numpy()
        front = _pareto_mask(x, y)
        if front.any():
            order = np.argsort(x[front])
            fig.add_trace(
                go.Scatter(
                    x=x[front][order],
                    y=y[front][order],
                    mode="lines+markers",
                    name="Pareto front",
                    line=dict(dash="dash", color="black"),
                    marker=dict(color="black", size=6),
                )
            )

    if label_top_n > 0 and label_col in df.columns:
        score = pd.to_numeric(df[y_metric], errors="coerce") / (
            pd.to_numeric(df[x_metric], errors="coerce").abs() + 1e-12
        )
        for index in score.nlargest(int(label_top_n)).index:
            fig.add_annotation(
                x=df.loc[index, x_metric],
                y=df.loc[index, y_metric],
                text=str(df.loc[index, label_col]),
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-20,
            )

    fig.update_layout(
        title=title or f"PCA trade-off: {y_metric} vs {x_metric}",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)


def plot_pca_metric_ranking(
    summary_df,
    metric,
    group_col="matrix_method",
    label_col="preprocessing",
    ascending=False,
    top_n=None,
    selected_col: str | None = None,
    title=None,
    width=1000,
    height=600,
    show=True,
):
    """Horizontal ranking, faceted by matrix family without redundant colours."""
    if metric not in summary_df.columns:
        raise ValueError(f"Metric {metric!r} not found.")
    df = summary_df.copy()
    groups = ordered_unique(df[group_col].astype(str)) if group_col in df.columns else ["all"]
    n_cols = len(groups)
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=groups, shared_yaxes=False)

    for col_index, group in enumerate(groups, start=1):
        sub = df if group == "all" else df[df[group_col].astype(str).eq(group)]
        sub = sub.sort_values(metric, ascending=ascending)
        if top_n is not None:
            sub = sub.head(int(top_n))
        # Reverse order for a top-to-bottom ranking in horizontal bars.
        sub = sub.iloc[::-1]
        marker_colors = None
        if selected_col is not None and selected_col in sub.columns:
            marker_colors = [
                "black" if bool(value) else "lightgray"
                for value in sub[selected_col].fillna(False)
            ]
        fig.add_trace(
            go.Bar(
                x=pd.to_numeric(sub[metric], errors="coerce"),
                y=sub[label_col].astype(str),
                orientation="h",
                marker_color=marker_colors,
                name=group,
                showlegend=False,
                hovertemplate=(
                    f"{label_col}: %{{y}}<br>{metric}: %{{x:.4f}}<extra></extra>"
                ),
            ),
            row=1,
            col=col_index,
        )
        fig.update_xaxes(title_text=metric, row=1, col=col_index)
        fig.update_yaxes(title_text=label_col, row=1, col=col_index)

    fig.update_layout(
        title=title or f"Ranking by {metric}",
        width=width,
        height=height,
    )
    apply_project_theme(fig)
    return show_or_return(fig, show)
