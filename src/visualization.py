"""
Plotly visualization module designed with Apple HIG principles:
- Clear typography (Inter)
- Apple semantic color palette (System Blue, Green, Orange, Red, Purple)
- Subtle gridlines and elevated surface backgrounds (#161F30)
- Purposeful visual hierarchy with minimal chart junk
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

APPLE_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#161F30",
    plot_bgcolor="#161F30",
    font=dict(
        color="#F8FAFC",
        family="Inter, -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
        size=12
    ),
    margin=dict(l=48, r=40, t=54, b=44),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color="#94A3B8")
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color="#94A3B8")
    )
)

APPLE_COLORS = [
    "#0A84FF",  # System Blue
    "#30D158",  # System Green
    "#FF9F0A",  # System Orange
    "#BF5AF2",  # System Purple
    "#64D2FF",  # System Cyan
    "#FF453A"   # System Red
]

def plot_model_comparison_bar(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    higher_is_better: bool = True
) -> go.Figure:
    """Creates a horizontal bar chart comparing models on a metric with Apple aesthetics."""
    df_sorted = df.sort_values(metric_col, ascending=higher_is_better)
    
    # Color scale: subtle gradient from deep blue to bright cyan
    fig = px.bar(
        df_sorted,
        x=metric_col,
        y="Algorithm",
        orientation="h",
        title=title,
        color=metric_col,
        color_continuous_scale="Blues" if higher_is_better else "Reds_r",
        text=metric_col
    )
    fig.update_layout(**APPLE_DARK_LAYOUT, showlegend=False, coloraxis_showscale=False)
    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
        marker_line_width=0,
        opacity=0.92
    )
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> go.Figure:
    """Scatter plot of actual vs predicted values with reference 1:1 line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(
            color="#0A84FF",
            size=6,
            opacity=0.65,
            line=dict(color="rgba(255, 255, 255, 0.2)", width=0.5)
        ),
        name="Test Prediction"
    ))
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="#FF453A", dash="dash", width=1.75),
        name="Ideal 1:1 Valuation"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Actual Price (₹)",
        yaxis_title="Predicted Price (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **APPLE_DARK_LAYOUT
    )
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> go.Figure:
    """Scatter plot of residuals against predicted values."""
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode="markers",
        marker=dict(
            color="#30D158",
            size=6,
            opacity=0.65,
            line=dict(color="rgba(255, 255, 255, 0.2)", width=0.5)
        ),
        name="Residuals"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#FF453A", line_width=1.5)
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Price (₹)",
        yaxis_title="Residual (Actual - Predicted) (₹)",
        **APPLE_DARK_LAYOUT
    )
    return fig


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], title: str) -> go.Figure:
    """Heatmap of confusion matrix with high-contrast text."""
    cm_arr = np.array(cm)
    fig = px.imshow(
        cm_arr,
        x=class_names,
        y=class_names,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        text_auto=True,
        title=title
    )
    fig.update_layout(**APPLE_DARK_LAYOUT, coloraxis_showscale=False)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart showing top feature importances."""
    top_df = importance_df.head(top_n).sort_values("Importance", ascending=True)
    fig = px.bar(
        top_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Predictive Feature Importances",
        color="Importance",
        color_continuous_scale="Purples",
        text="Importance"
    )
    fig.update_layout(**APPLE_DARK_LAYOUT, showlegend=False, coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", opacity=0.92)
    return fig


def plot_pca_clusters(pca_df: pd.DataFrame, cluster_col: str, title: str) -> go.Figure:
    """2D PCA projection scatter plot colored by cluster."""
    fig = px.scatter(
        pca_df,
        x="PCA_1",
        y="PCA_2",
        color=pca_df[cluster_col].astype(str),
        hover_data=["ram", "weight", "inches", "storage_gb", "ppi"],
        title=title,
        color_discrete_sequence=APPLE_COLORS
    )
    fig.update_layout(
        legend_title="Cluster",
        xaxis_title="Principal Component 1 (Primary Variance)",
        yaxis_title="Principal Component 2 (Secondary Variance)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **APPLE_DARK_LAYOUT
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    return fig


def plot_tsne_clusters(tsne_df: pd.DataFrame, cluster_col: str, title: str) -> go.Figure:
    """2D t-SNE projection scatter plot colored by cluster."""
    fig = px.scatter(
        tsne_df,
        x="tSNE_1",
        y="tSNE_2",
        color=tsne_df[cluster_col].astype(str),
        hover_data=["ram", "weight", "inches", "storage_gb", "ppi"],
        title=title,
        color_discrete_sequence=APPLE_COLORS
    )
    fig.update_layout(
        legend_title="Cluster",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **APPLE_DARK_LAYOUT
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    return fig


def plot_clustering_diagnostics(
    k_values: list[int],
    inertias: list[float],
    silhouettes: list[float],
    db_indices: list[float],
    ch_indices: list[float]
) -> tuple[go.Figure, go.Figure]:
    """Generates elbow and silhouette/Davies-Bouldin diagnostic figures."""
    # Elbow figure
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=k_values,
        y=inertias,
        mode="lines+markers",
        marker=dict(size=7, color="#0A84FF"),
        line=dict(width=2.2, color="#0A84FF"),
        name="Inertia"
    ))
    fig_elbow.update_layout(
        title="K-Means Elbow Method (Inertia vs k)",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Sum of Squared Errors (Inertia)",
        **APPLE_DARK_LAYOUT
    )

    # Multi-metric score figure
    fig_scores = go.Figure()
    fig_scores.add_trace(go.Scatter(
        x=k_values,
        y=silhouettes,
        mode="lines+markers",
        marker=dict(size=7, color="#30D158"),
        line=dict(width=2, color="#30D158"),
        name="Silhouette Score (Higher is better)"
    ))
    fig_scores.add_trace(go.Scatter(
        x=k_values,
        y=db_indices,
        mode="lines+markers",
        marker=dict(size=7, color="#FF9F0A"),
        line=dict(width=2, color="#FF9F0A"),
        name="Davies-Bouldin Index (Lower is better)"
    ))
    fig_scores.update_layout(
        title="Cluster Validation Metrics across k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Metric Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **APPLE_DARK_LAYOUT
    )

    return fig_elbow, fig_scores


def plot_probabilities_bar(prob_dict: dict[str, float], title: str) -> go.Figure:
    """Horizontal bar chart for classification predicted class probabilities."""
    df = pd.DataFrame(list(prob_dict.items()), columns=["Category", "Probability"])
    fig = px.bar(
        df,
        x="Probability",
        y="Category",
        orientation="h",
        title=title,
        color="Category",
        text="Probability",
        color_discrete_sequence=["#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2"]
    )
    fig.update_layout(xaxis_title="Model Confidence (%)", showlegend=False, **APPLE_DARK_LAYOUT)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", opacity=0.92)
    return fig


def plot_class_distribution(series: pd.Series, title: str) -> go.Figure:
    """Bar chart for target class balance."""
    df = series.value_counts().reset_index()
    df.columns = ["Class", "Count"]
    fig = px.bar(
        df,
        x="Class",
        y="Count",
        title=title,
        color="Class",
        text="Count",
        color_discrete_sequence=APPLE_COLORS
    )
    fig.update_layout(showlegend=False, **APPLE_DARK_LAYOUT)
    fig.update_traces(textposition="outside", opacity=0.92)
    return fig
