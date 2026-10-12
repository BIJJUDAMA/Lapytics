import json
import streamlit as st
import pandas as pd
from src.config import  COMPANIES, TYPENAMES, OPSYS_LIST, RESULTS_DIR
from src.model_registry import get_available_models
from src.inference import predict_price_category
from src.data_loader import load_classification_metrics, load_classification_data
from src.visualization import (
    plot_probabilities_bar,
    plot_model_comparison_bar,
    plot_confusion_matrix,
    plot_class_distribution
)
from src.ui import (
    apply_custom_css,
    render_page_header,
    render_metric_card,
    render_section_header
)

st.set_page_config(
    page_title="Price Classification | Laptop Intelligence",
    layout="wide"
)

apply_custom_css()

render_page_header(
    title="Laptop Price Classification",
    subtitle="Categorize laptop hardware profiles into four discrete market segments: Budget, Mid-range, Premium, or Flagship.",
    icon_name="layers",
    eyebrow="Market Tier Classification"
)

tab_predict, tab_compare = st.tabs(["Predict Market Tier", "Algorithm Leaderboard & Confusion Matrices"])

with tab_predict:
    available_clfs = get_available_models("classification")
    if not available_clfs:
        st.error("No classification models found in `models/classification/`. Please run `scripts/export_models.py`.")
        st.stop()

    render_section_header("Classifier Pipeline Selection", icon_name="cpu")
    selected_clf = st.selectbox("Select Fitted Classifier", available_clfs, index=0)

    render_section_header("Hardware Specifications", "Configure machine specs to evaluate segment classification", icon_name="laptop")

    COMMON_RESOLUTIONS = [
        (1920, 1080, "Full HD (1920 x 1080)"),
        (1366, 768, "HD (1366 x 768)"),
        (2560, 1440, "QHD (2560 x 1440)"),
        (3840, 2160, "4K UHD (3840 x 2160)"),
        (2560, 1600, "Retina / WQXGA (2560 x 1600)"),
        (2880, 1800, "Retina 15\" (2880 x 1800)")
    ]

    with st.form("classification_form", border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Identity & Platform**")
            company = st.selectbox("Brand / Manufacturer", COMPANIES, index=COMPANIES.index("Dell") if "Dell" in COMPANIES else 0)
            typename = st.selectbox("Laptop Type", TYPENAMES, index=TYPENAMES.index("Notebook") if "Notebook" in TYPENAMES else 0)
            opsys = st.selectbox("Operating System", OPSYS_LIST, index=OPSYS_LIST.index("Windows 10") if "Windows 10" in OPSYS_LIST else 0)

        with c2:
            st.markdown("**Memory & Chassis**")
            ram = st.select_slider("RAM (GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
            weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
            inches = st.number_input("Screen Size (inches)", min_value=10.0, max_value=18.5, value=15.6, step=0.1)

        with c3:
            st.markdown("**Display & Storage**")
            res_idx = st.selectbox("Screen Resolution", range(len(COMMON_RESOLUTIONS)), format_func=lambda i: COMMON_RESOLUTIONS[i][2], index=0)
            storage_gb = st.select_slider("Storage Capacity (GB)", options=[32, 64, 128, 256, 512, 1024, 2048], value=256)
            
            st.write("")
            cb_col1, cb_col2 = st.columns(2)
            with cb_col1:
                has_ssd = st.checkbox("SSD Included", value=True)
                has_hdd = st.checkbox("HDD Included", value=False)
            with cb_col2:
                is_ips = st.checkbox("IPS Display", value=False)
                is_touch = st.checkbox("Touchscreen", value=False)

        submit_clf = st.form_submit_button("Classify Market Tier", type="primary", use_container_width=True)

    res_w, res_h, _ = COMMON_RESOLUTIONS[res_idx]

    laptop_specs = {
        "company": company,
        "typename": typename,
        "opsys": opsys,
        "ram": float(ram),
        "weight": float(weight),
        "inches": float(inches),
        "res_width": int(res_w),
        "res_height": int(res_h),
        "storage_gb": float(storage_gb),
        "has_ssd": 1 if has_ssd else 0,
        "has_hdd": 1 if has_hdd else 0,
        "is_ips": 1 if is_ips else 0,
        "is_touchscreen": 1 if is_touch else 0,
    }

    if submit_clf:
        try:
            with st.spinner("Evaluating multi-class model..."):
                res = predict_price_category(selected_clf, laptop_specs)

            with st.container(border=True):
                res_c1, res_c2, res_c3 = st.columns(3)
                with res_c1:
                    render_metric_card(
                        label="Predicted Category",
                        value=res["predicted_category"],
                        subtext="Target Market Segment",
                        icon_name="tag"
                    )
                with res_c2:
                    acc_str = f"{res['accuracy'] * 100:.1f}%" if res['accuracy'] is not None else "N/A"
                    render_metric_card(
                        label="Model Test Accuracy",
                        value=acc_str,
                        subtext=f"Pipeline: {selected_clf}",
                        icon_name="target"
                    )
                with res_c3:
                    f1_str = f"{res['weighted_f1']:.4f}" if res['weighted_f1'] is not None else "N/A"
                    render_metric_card(
                        label="Model Weighted F1",
                        value=f1_str,
                        subtext="Multi-Class Balance",
                        icon_name="scale"
                    )

                if res["probabilities"]:
                    st.markdown("#### Class Confidence Probabilities")
                    fig_prob = plot_probabilities_bar(res["probabilities"], "Predicted Class Confidence Distribution")
                    st.plotly_chart(fig_prob, use_container_width=True)
                else:
                    st.info("Class probability estimates are unavailable for non-probabilistic models.")

        except Exception as e:
            st.error(f"Classification Error: {str(e)}")

with tab_compare:
    render_section_header("10-Algorithm Classification Leaderboard", icon_name="bar-chart-2")
    clf_metrics_df = load_classification_metrics()

    if clf_metrics_df.empty:
        st.warning("No metrics found in `results/classification_metrics.csv`.")
    else:
        st.dataframe(clf_metrics_df, use_container_width=True)

        fig_comp = plot_model_comparison_bar(clf_metrics_df, "Accuracy", "Holdout Test Accuracy Across 10 Classifiers")
        st.plotly_chart(fig_comp, use_container_width=True)

        col_cm, col_dist = st.columns(2)

        with col_cm:
            st.subheader("Interactive Confusion Matrix")
            cm_model = st.selectbox("Inspect Confusion Matrix for Model", clf_metrics_df["Algorithm"].tolist(), index=0)
            cm_file = RESULTS_DIR / "classification_confusion_matrices.json"
            if cm_file.exists():
                with open(cm_file, "r") as f:
                    cm_data = json.load(f)
                cms = cm_data.get("confusion_matrices", {})
                if cm_model in cms:
                    fig_cm = plot_confusion_matrix(cms[cm_model], cm_data["classes"], f"Confusion Matrix: {cm_model}")
                    st.plotly_chart(fig_cm, use_container_width=True)

        with col_dist:
            st.subheader("Ground Truth Class Distribution")
            df_clf = load_classification_data()
            fig_dist = plot_class_distribution(df_clf["price_category"], "Distribution across 1,243 Cleaned Records")
            st.plotly_chart(fig_dist, use_container_width=True)
