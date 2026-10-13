import streamlit as st
import pandas as pd
from src.data_loader import load_regression_metrics, load_classification_metrics, load_clustering_metrics
from src.ui import (
    apply_custom_css,
    render_page_header,
    render_metric_card,
    render_section_header,
    render_spec_card,
    lucide_icon
)

st.set_page_config(
    page_title="Laptop Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Header
render_page_header(
    title="Laptop Intelligence Dashboard",
    subtitle="Interactive Machine Learning Capstone: Multi-Algorithm Valuation, Market Tier Classification & Latent Segmentation",
    icon_name="laptop",
    eyebrow="Machine Learning Capstone"
)

# High-Level KPI Cards
reg_df = load_regression_metrics()
best_reg_r2 = f"{reg_df.iloc[0]['R2']:.4f}" if not reg_df.empty else "0.8121"

best_reg_rmse = f"₹{reg_df.iloc[0]['RMSE']:,.0f}" if not reg_df.empty else "₹14,869"

clf_df = load_classification_metrics()
best_clf_acc = f"{clf_df.iloc[0]['Accuracy'] * 100:.1f}%" if not clf_df.empty else "70.7%"

clust_metrics = load_clustering_metrics()
k_clusters = len(clust_metrics.get("cluster_profiles", [])) or 4

c1, c2, c3, c4 = st.columns(4)

with c1:
    render_metric_card(
        label="Top Regression R²",
        value=best_reg_r2,
        subtext="Gradient Boosting (Tuned)",
        icon_name="trending-up"
    )

with c2:
    render_metric_card(
        label="Top Regression RMSE",
        value=best_reg_rmse,
        subtext="Root Mean Squared Error",
        icon_name="target"
    )

with c3:
    render_metric_card(
        label="Top Classifier Accuracy",
        value=best_clf_acc,
        subtext="Random Forest (Tuned)",
        icon_name="tag"
    )

with c4:
    render_metric_card(
        label="Market Segments",
        value=f"{k_clusters} Archetypes",
        subtext="K-Means Unsupervised",
        icon_name="grid"
    )

# Machine Learning Modules
render_section_header(
    title="Core Intelligence Modules",
    subtitle="Navigate to the specialized interactive machine learning interfaces below",
    icon_name="cpu"
)

mod1, mod2, mod3 = st.columns(3)

with mod1:
    with st.container(border=True):
        st.markdown(f"#### {lucide_icon('tag', size=18, color='#38BDF8')} Price Predictor", unsafe_allow_html=True)
        st.write("Continuous laptop valuation in Indian Rupees (₹) with real-time specification validation and side-by-side spec comparison.")
        st.page_link("pages/1_Price_Predictor.py", label="Open Price Predictor →")

with mod2:
    with st.container(border=True):
        st.markdown(f"#### {lucide_icon('layers', size=18, color='#34D399')} Price Classification", unsafe_allow_html=True)
        st.write("4-tier price category prediction (Budget, Mid-range, Premium, Flagship) with class probability confidence breakdowns.")
        st.page_link("pages/2_Price_Classification.py", label="Open Price Classification →")

with mod3:
    with st.container(border=True):
        st.markdown(f"#### {lucide_icon('grid', size=18, color='#C084FC')} Laptop Clustering", unsafe_allow_html=True)
        st.write("Unsupervised discovery of hardware groupings using K-Means and Agglomerative Clustering with 2D PCA & t-SNE projections.")
        st.page_link("pages/3_Laptop_Clustering.py", label="Open Laptop Clustering →")

# Model Laboratory Callout
render_section_header(
    title="Comprehensive Model Laboratory",
    subtitle="Evaluate diagnostic residual plots, feature importances, confusion matrices, and benchmark comparisons",
    icon_name="flask-conical"
)

with st.container(border=True):
    lab_col1, lab_col2 = st.columns([3, 1])
    with lab_col1:
        st.markdown(f"#### {lucide_icon('flask-conical', size=18, color='#FBBF24')} Benchmarking & Architectural Transparency", unsafe_allow_html=True)
        st.write("Inspect all 10 regression algorithms, 10 classification models, cross-validation stability, diagnostic residual analysis, and multi-class confusion matrices.")
    with lab_col2:
        st.write("")
        st.page_link("pages/4_Model_Laboratory.py", label="Launch Model Lab →")

# Dataset Details & Limitations
render_section_header("Data Governance & Model Boundaries", icon_name="shield-check")

col_left, col_right = st.columns(2)

with col_left:
    render_spec_card(
        title="Dataset Architecture",
        icon_name="database",
        icon_color="#60A5FA",
        items=[
            ("Sample Records", "1,243 deduplicated"),
            ("Input Dimension", "42 encoded features"),
            ("Engineered Metrics", "PPI, Storage GB, SSD/HDD"),
            ("Train / Test Split", "80% / 20% (stratified, seed 42)")
        ]
    )

with col_right:
    render_spec_card(
        title="Prediction Boundaries",
        icon_name="alert-triangle",
        icon_color="#FBBF24",
        items=[
            ("Valuation Currency", "Indian Rupees (₹ INR)"),
            ("Target Range", "₹9,270 – ₹1,50,358 (IQR)"),
            ("Hardware Scope", "Retail catalog specifications"),
            ("Target Leakage", "Zero price exposure (strict)")
        ]
    )
