from __future__ import annotations

from typing import Sequence
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.utils import wavelength_axis
from src.visualization.common import show_or_return
from src.visualization.plot_scores import plot_scores
from src.visualization.plot_diagnostics import plot_metric_by_index, plot_xy_diagnostic

def plot_explained_variance(
    explained_variance_ratio,
    cumulative_explained_variance_ratio=None,
    n_components_to_show: int | None = None,
    title: str = "Explained variance",
    width: int = 850,
    height: int = 500,
    show: bool = True,
):
    evr = np.asarray(explained_variance_ratio, dtype=float)
    cum = np.cumsum(evr) if cumulative_explained_variance_ratio is None else np.asarray(cumulative_explained_variance_ratio, dtype=float)
    if n_components_to_show is not None:
        evr = evr[:n_components_to_show]
        cum = cum[:n_components_to_show]
    pcs = np.arange(1, len(evr) + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pcs, y=evr, name="Explained variance", hovertemplate="PC%{x}<br>variance: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=pcs, y=cum, mode="lines+markers", name="Cumulative variance", hovertemplate="PC%{x}<br>cumulative: %{y:.4f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Component", yaxis_title="Variance ratio", width=width, height=height)
    return show_or_return(fig, show)


def plot_loadings(
    loadings: np.ndarray,
    #pca_res=None,
    wavelengths=None,
    components: Sequence[int] = (1, 2, 3),
    component_names: Sequence[str] | None = None,
    title: str = "Loadings",
    width: int = 900,
    height: int = 500,
    show: bool = True,
):
    x, x_title = wavelength_axis(loadings.shape[0], wavelengths)
    fig = go.Figure()
    for k, comp in enumerate(components):
        j = comp - 1
        if j >= loadings.shape[1]:
            continue
        name = component_names[k] if component_names is not None and k < len(component_names) else f"PC{comp}"
        fig.add_trace(go.Scatter(x=x, y=loadings[:, j], mode="lines+markers", name=name))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title="Loading", width=width, height=height)
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
        title=title or f"Biplot: C{dims[0]} vs C{dims[1]}", 
        show=False, 
        **metadata
    )
    ix, iy = dims[0] - 1, dims[1] - 1
    strength = np.sqrt(loadings[:, ix] ** 2 + loadings[:, iy] ** 2)
    top_idx = np.argsort(strength)[-n_loadings:]
    score_range = max(np.nanmax(np.abs(scores[:, ix])), np.nanmax(np.abs(scores[:, iy])))
    for j in top_idx:
        x_end = loadings[j, ix] * score_range * loading_scale
        y_end = loadings[j, iy] * score_range * loading_scale
        label = f"band {j}" if wavelengths is None else f"{np.asarray(wavelengths)[j]:.1f} nm"
        fig.add_annotation(x=x_end, y=y_end, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, text=label)
    return show_or_return(fig, show)


def plot_pca_metric_t2(pca_model, X=None, labels=None, object_ids=None, source_images=None, n_components=None, title="Hotelling T²", show=True):
    T2 = pca_model.hotelling_t2(X, n_components=n_components)
    return plot_metric_by_index(T2, labels=labels, object_ids=object_ids, source_images=source_images, title=title, y_title="Hotelling T²", show=show)

def plot_pca_metric_q(pca_model, X=None, labels=None, object_ids=None, source_images=None, n_components=None, title="Q residuals", show=True):
    if X is None:
        X = pca_model.inverse_transform(pca_model.scores_)
    Q, _ = pca_model.q_residuals(X, n_components=n_components)
    return plot_metric_by_index(Q, labels=labels, object_ids=object_ids, source_images=source_images, title=title, y_title="Q residual", show=show)


def plot_pca_diagnostic(pca_model, X=None, labels=None, object_ids=None, source_images=None, n_components=None, title="PCA diagnostic: Q residuals vs Hotelling T²", show=True):
    if X is None:
        X = pca_model.inverse_transform(pca_model.scores_)
    T2, Q = pca_model.distances(X, n_components=n_components)
    return plot_xy_diagnostic(T2, Q, labels=labels, object_ids=object_ids, source_images=source_images, title=title, x_title="Hotelling T²", y_title="Q residual", show=show)


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
):
    """
    Plot a heatmap for one PCA comparison metric.

    Useful to compare preprocessing x matrix representation.
    """
    if metric not in summary_df.columns:
        raise ValueError(f"Metric '{metric}' not found in summary_df.")

    table = summary_df.pivot_table(
        index=index_col,
        columns=column_col,
        values=metric,
        aggfunc="mean",
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=table.values,
            x=table.columns.astype(str),
            y=table.index.astype(str),
            colorscale=colorscale,
            colorbar=dict(title=metric),
            hovertemplate=(
                f"{column_col}: %{{x}}<br>"
                f"{index_col}: %{{y}}<br>"
                f"{metric}: %{{z:.4f}}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or f"PCA metric heatmap — {metric}",
        xaxis_title=column_col,
        yaxis_title=index_col,
        width=width,
        height=height,
    )

    return show_or_return(fig, show)


def plot_pca_metric_tradeoff(
    summary_df,
    x_metric="batch_trace_ratio",
    y_metric="class_trace_ratio",
    color_by="matrix_method",
    symbol_by="preprocessing",
    size_by=None,
    hover_cols=None,
    title=None,
    width=900,
    height=650,
    show=True,
):
    """
    Plot trade-off between two PCA metrics.

    Typical use:
        x = batch effect
        y = class separation
    """
    if x_metric not in summary_df.columns:
        raise ValueError(f"x_metric '{x_metric}' not found.")

    if y_metric not in summary_df.columns:
        raise ValueError(f"y_metric '{y_metric}' not found.")

    if hover_cols is None:
        hover_cols = [
            col for col in [
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
            ]
            if col in summary_df.columns
        ]

    fig = px.scatter(
        summary_df,
        x=x_metric,
        y=y_metric,
        color=color_by if color_by in summary_df.columns else None,
        symbol=symbol_by if symbol_by in summary_df.columns else None,
        size=size_by if size_by in summary_df.columns else None,
        hover_data=hover_cols,
        title=title or f"PCA trade-off: {y_metric} vs {x_metric}",
    )

    fig.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(width=1, color="black"),
        )
    )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title=x_metric,
        yaxis_title=y_metric,
    )

    return show_or_return(fig, show)


def plot_pca_metric_ranking(
    summary_df,
    metric,
    group_col="matrix_method",
    label_col="preprocessing",
    ascending=False,
    top_n=None,
    title=None,
    width=1000,
    height=600,
    show=True,
):
    """
    Plot ranking of preprocessing methods according to one metric.
    """
    if metric not in summary_df.columns:
        raise ValueError(f"Metric '{metric}' not found.")

    df = summary_df.copy()

    df = df.sort_values(metric, ascending=ascending)

    if top_n is not None:
        df = (
            df.groupby(group_col, group_keys=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    fig = px.bar(
        df,
        x=label_col,
        y=metric,
        color=group_col if group_col in df.columns else None,
        facet_col=group_col if group_col in df.columns else None,
        hover_data=[
            col for col in [
                "matrix_method",
                "preprocessing",
                "class_trace_ratio",
                "batch_trace_ratio",
                "class_over_batch_ratio",
                "ncomp_90",
                "ncomp_95",
            ]
            if col in df.columns
        ],
        title=title or f"Ranking by {metric}",
    )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title=label_col,
        yaxis_title=metric,
    )

    fig.update_xaxes(tickangle=45)

    return show_or_return(fig, show)

