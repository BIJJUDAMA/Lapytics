import json
import streamlit as st
import pandas as pd
from src.config import RESULTS_DIR
from src.data_loader import load_clustering_metrics, load_clustering_projections
from src.visualization import (
    plot_pca_clusters,
    plot_tsne_clusters,
    plot_clustering_diagnostics
)
from src.ui import (
    apply_custom_css,
    render_page_header,
    render_metric_card,
    render_section_header
)

st.set_page_config(
    page_title="Laptop Clustering | Laptop Intelligence",
    layout="wide"
)

apply_custom_css()

render_page_header(
    title="Laptop Clustering & Market Segmentation",
    subtitle="Unsupervised pattern recognition discovering natural laptop hardware architectures without supervisory labels.",
    icon_name="grid",
    eyebrow="Unsupervised Learning"
)

clust_metrics = load_clustering_metrics()
if not clust_metrics:
    st.error("Clustering metrics not found in `results/clustering_metrics.json`. Please run `python scripts/prepare_clustering.py`.")
    st.stop()

proj_df = load_clustering_projections()

@st.fragment
def render_clustering_interactive_section(proj_df: pd.DataFrame, clust_metrics: dict):
    render_section_header("Algorithm & Partition Configuration", icon_name="cpu")

    with st.container(border=True):
        c_alg, c_k = st.columns(2)
        with c_alg:
            algorithm = st.selectbox("Clustering Method", ["K-Means", "Agglomerative Hierarchical (Ward Linkage)"])
        with c_k:
            k = st.slider("Number of Clusters (k)", min_value=2, max_value=6, value=4)

    cluster_col = f"kmeans_cluster_{k}" if algorithm == "K-Means" else f"agg_cluster_{k}"

    render_section_header(
        title="Manifold Projections (PCA & t-SNE)",
        subtitle="Interactive two-dimensional projections preserving dimensional variance and local geometry",
        icon_name="activity"
    )

    if not proj_df.empty:
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig_pca = plot_pca_clusters(proj_df, cluster_col, f"2D PCA Projection ({algorithm}, k={k})")
            st.plotly_chart(fig_pca, use_container_width=True)
            exp_var = clust_metrics.get("pca_explained_variance_ratio", [0, 0])
            st.caption(f"Principal Component 1: {exp_var[0]*100:.1f}% variance | Principal Component 2: {exp_var[1]*100:.1f}% variance.")

        with v_col2:
            fig_tsne = plot_tsne_clusters(proj_df, cluster_col, f"2D t-SNE Manifold ({algorithm}, k={k})")
            st.plotly_chart(fig_tsne, use_container_width=True)
            st.caption("t-SNE non-linear projection highlighting cluster density and neighborhood groupings.")
    else:
        st.warning("Projection coordinates not found.")

    render_section_header(
        title="Discovered Cluster Profiles",
        subtitle="Hardware specification averages and automated descriptive summaries per segment",
        icon_name="layers"
    )

    profiles = clust_metrics.get("cluster_profiles", [])
    if profiles and k == 4 and algorithm == "K-Means":
        prof_df = pd.DataFrame(profiles)
        st.dataframe(prof_df, use_container_width=True)

        cols = st.columns(len(profiles))
        for idx, (col, prof) in enumerate(zip(cols, profiles)):
            with col:
                with st.container(border=True):
                    st.markdown(f"**Cluster {prof['Cluster']}**")
                    st.markdown(f"### {prof['Count']} <small style='font-size:0.85rem;color:#94A3B8;'>({prof['Percentage']}%)</small>", unsafe_allow_html=True)
                    st.markdown(f"""- **RAM**: {prof['Avg_RAM_GB']} GB
- **Weight**: {prof['Avg_Weight_kg']} kg
- **Storage**: {prof['Avg_Storage_GB']:.0f} GB
- **SSD**: {prof['SSD_Percentage']}%
- **PPI**: {prof['Avg_PPI']}""")
                    st.caption(prof['Summary'])
    else:
        if not proj_df.empty:
            dyn_df = proj_df.groupby(cluster_col).agg(
                Count=("ram", "count"),
                Avg_RAM=("ram", "mean"),
                Avg_Weight=("weight", "mean"),
                Avg_Storage=("storage_gb", "mean"),
                Avg_PPI=("ppi", "mean")
            ).reset_index()
            dyn_df["Percentage"] = (dyn_df["Count"] / len(proj_df) * 100).round(1)
            st.dataframe(dyn_df, use_container_width=True)

render_clustering_interactive_section(proj_df, clust_metrics)

# Diagnostic Curves & Validation
render_section_header(
    title="Cluster Validation Diagnostics",
    subtitle="Quantitative metrics for determining optimal cluster count (k) and linkage strategy",
    icon_name="bar-chart-2"
)

k_vals = clust_metrics.get("k_range", [])
inertias = clust_metrics.get("kmeans_elbow", [])
silhouettes = clust_metrics.get("silhouette_scores", [])
db_indices = clust_metrics.get("davies_bouldin_scores", [])
ch_indices = clust_metrics.get("calinski_harabasz_scores", [])

fig_elbow, fig_scores = plot_clustering_diagnostics(k_vals, inertias, silhouettes, db_indices, ch_indices)

d_col1, d_col2 = st.columns(2)
with d_col1:
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.caption("The Elbow method illustrates marginal inertia reduction tapering off beyond k=4.")

with d_col2:
    st.plotly_chart(fig_scores, use_container_width=True)
    st.caption("Higher Silhouette / Calinski-Harabasz and lower Davies-Bouldin indicate superior cluster separation.")

st.markdown("#### Hierarchical Linkage Comparison")
linkages = clust_metrics.get("linkage_comparison", [])
if linkages:
    link_df = pd.DataFrame(linkages)
    st.dataframe(link_df, use_container_width=True)
