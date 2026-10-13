import json
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, RANDOM_STATE, RESULTS_DIR
from src.data_loader import (
    load_regression_metrics,
    load_classification_metrics,
    load_clustering_metrics,
    load_regression_data
)
from src.model_registry import load_model, get_available_models
from src.visualization import (
    plot_model_comparison_bar,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_confusion_matrix
)
from src.ui import (
    apply_custom_css,
    render_page_header,
    render_section_header
)

st.set_page_config(
    page_title="Model Laboratory | Laptop Intelligence",
    layout="wide"
)

apply_custom_css()

render_page_header(
    title="Model Laboratory",
    subtitle="Full architectural benchmarking, test set diagnostics, and confusion matrices.",
    icon_name="flask-conical",
    eyebrow="Evaluation & Diagnostics Center"
)

tab_reg, tab_clf, tab_clust = st.tabs([
    "Regression Benchmarks",
    "Classification Benchmarks",
    "Clustering Diagnostics"
])

# ----------------- REGRESSION TAB -----------------
with tab_reg:
    render_section_header("10-Algorithm Regression Leaderboard", icon_name="trending-up")
    reg_df = load_regression_metrics()
    if reg_df.empty:
        st.warning("No regression metrics available.")
    else:
        st.dataframe(reg_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_r2 = plot_model_comparison_bar(reg_df, "R2", "R² Score Benchmark (Higher is Better)")
            st.plotly_chart(fig_r2, use_container_width=True)
        with c2:
            fig_rmse = plot_model_comparison_bar(reg_df, "RMSE", "RMSE Metric Benchmark (Lower is Better)", higher_is_better=False)
            st.plotly_chart(fig_rmse, use_container_width=True)

        render_section_header("Regression Model Diagnostics (Holdout Test Set: 249 Records)", icon_name="activity")

        avail_reg = get_available_models("regression")
        selected_model_name = st.selectbox("Inspect Diagnostic Plots for Pipeline", avail_reg, index=0)

        @st.cache_data
        def get_cached_regression_diagnostics(model_name: str):
            df_reg = load_regression_data()
            X = df_reg.drop(columns=["price"]).copy()
            y = df_reg["price"].copy()
            target_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=target_bins)
            model = load_model("regression", model_name)
            y_pred = model.predict(X_test)
            importances = None
            if hasattr(model, "feature_importances_"):
                importances = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)
            return y_test.values, y_pred, importances

        try:
            y_test_vals, y_pred, importances = get_cached_regression_diagnostics(selected_model_name)

            diag_c1, diag_c2 = st.columns(2)
            with diag_c1:
                fig_avp = plot_actual_vs_predicted(y_test_vals, y_pred, f"Actual vs Predicted: {selected_model_name}")
                st.plotly_chart(fig_avp, use_container_width=True)
            with diag_c2:
                fig_res = plot_residuals(y_test_vals, y_pred, f"Residuals Plot: {selected_model_name}")
                st.plotly_chart(fig_res, use_container_width=True)

            if importances is not None:
                st.markdown("#### Feature Importance Analysis")
                fig_imp = plot_feature_importance(importances, top_n=15)
                st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Error computing diagnostics: {str(e)}")


# ----------------- CLASSIFICATION TAB -----------------
with tab_clf:
    render_section_header("10-Algorithm Classification Leaderboard", icon_name="layers")
    clf_df = load_classification_metrics()
    if clf_df.empty:
        st.warning("No classification metrics available.")
    else:
        st.dataframe(clf_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_acc = plot_model_comparison_bar(clf_df, "Accuracy", "Accuracy Benchmark Across Classifiers")
            st.plotly_chart(fig_acc, use_container_width=True)
        with c2:
            fig_f1 = plot_model_comparison_bar(clf_df, "Weighted_F1", "Weighted F1-Score Across Classifiers")
            st.plotly_chart(fig_f1, use_container_width=True)

        render_section_header("Multi-Class Confusion Matrix Inspection", icon_name="bar-chart-2")
        cm_file = RESULTS_DIR / "classification_confusion_matrices.json"
        if cm_file.exists():
            with open(cm_file, "r") as f:
                cm_data = json.load(f)
            cms = cm_data.get("confusion_matrices", {})
            cm_sel = st.selectbox("Select Model to View Confusion Matrix", list(cms.keys()), index=0)
            if cm_sel in cms:
                fig_cm = plot_confusion_matrix(cms[cm_sel], cm_data["classes"], f"Confusion Matrix: {cm_sel}")
                st.plotly_chart(fig_cm, use_container_width=True)


# ----------------- CLUSTERING TAB -----------------
with tab_clust:
    render_section_header("Unsupervised Clustering Diagnostics across k (2 to 10)", icon_name="grid")
    clust_metrics = load_clustering_metrics()
    if clust_metrics:
        k_df = pd.DataFrame({
            "k": clust_metrics["k_range"],
            "Inertia": clust_metrics["kmeans_elbow"],
            "Silhouette_Score": clust_metrics["silhouette_scores"],
            "Davies_Bouldin_Index": clust_metrics["davies_bouldin_scores"],
            "Calinski_Harabasz_Index": clust_metrics["calinski_harabasz_scores"]
        })
        st.dataframe(k_df, use_container_width=True)

        st.markdown("#### Agglomerative Hierarchical Linkage Comparison")
        link_df = pd.DataFrame(clust_metrics.get("linkage_comparison", []))
        st.dataframe(link_df, use_container_width=True)
