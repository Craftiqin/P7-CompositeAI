"""Interactive Plotly visualization builders."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def missing_value_matrix(data: pd.DataFrame) -> go.Figure:
    """Create missing-value matrix heatmap."""
    sample = data.head(500).isna().astype(int)
    fig = px.imshow(
        sample.T,
        aspect="auto",
        color_continuous_scale=["#f7fbff", "#d62828"],
        labels={"x": "Row", "y": "Feature", "color": "Missing"},
        title="Missing Value Matrix",
    )
    fig.update_layout(template="plotly_white", height=420)
    return fig


def correlation_heatmap(data: pd.DataFrame) -> go.Figure:
    """Create numeric correlation heatmap."""
    numeric = data.select_dtypes(include="number")
    correlation = numeric.corr(numeric_only=True).fillna(0)
    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
    )
    fig.update_layout(template="plotly_white", height=520)
    return fig


def distribution_plot(data: pd.DataFrame, column: str) -> go.Figure:
    """Create histogram distribution plot."""
    fig = px.histogram(data, x=column, marginal="box", title=f"Distribution: {column}")
    fig.update_layout(template="plotly_white", height=420)
    return fig


def boxplot(data: pd.DataFrame, column: str) -> go.Figure:
    """Create boxplot."""
    fig = px.box(data, y=column, points="outliers", title=f"Boxplot: {column}")
    fig.update_layout(template="plotly_white", height=420)
    return fig


def violin_plot(data: pd.DataFrame, column: str) -> go.Figure:
    """Create violin plot."""
    fig = px.violin(data, y=column, box=True, points="outliers", title=f"Violin: {column}")
    fig.update_layout(template="plotly_white", height=420)
    return fig


def scatterplot(data: pd.DataFrame, x_column: str, y_column: str, color: str | None = None) -> go.Figure:
    """Create scatterplot."""
    fig = px.scatter(data, x=x_column, y=y_column, color=color, title=f"{x_column} vs {y_column}")
    fig.update_layout(template="plotly_white", height=440)
    return fig


def scatter_matrix(data: pd.DataFrame, columns: list[str]) -> go.Figure:
    """Create scatter matrix for selected numeric columns."""
    fig = px.scatter_matrix(data, dimensions=columns[:6], title="Scatter Matrix")
    fig.update_layout(template="plotly_white", height=650)
    return fig


def category_counts(data: pd.DataFrame, column: str) -> go.Figure:
    """Create category count bar chart."""
    counts = data[column].value_counts(dropna=False).head(30).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count", title=f"Category Counts: {column}")
    fig.update_layout(template="plotly_white", height=420)
    return fig


def target_correlation(data: pd.DataFrame, target_column: str) -> go.Figure:
    """Create target correlation bar chart."""
    numeric = data.select_dtypes(include="number")
    if target_column not in numeric.columns:
        return go.Figure()
    correlations = numeric.corr(numeric_only=True)[target_column].drop(target_column).dropna()
    frame = correlations.sort_values(key=lambda values: values.abs(), ascending=False).reset_index()
    frame.columns = ["feature", "correlation"]
    fig = px.bar(frame, x="feature", y="correlation", title=f"Target Correlation: {target_column}")
    fig.update_layout(template="plotly_white", height=430)
    return fig


def outlier_scatter(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    outlier_mask: pd.Series,
) -> go.Figure:
    """Create outlier-highlighted scatterplot."""
    plot_data = data.copy()
    plot_data["outlier"] = outlier_mask.map({True: "Outlier", False: "Normal"})
    fig = px.scatter(
        plot_data,
        x=x_column,
        y=y_column,
        color="outlier",
        title=f"Outlier Scatter: {x_column} vs {y_column}",
    )
    fig.update_layout(template="plotly_white", height=440)
    return fig
