import streamlit as st
import pandas as pd
from src.config import COMPANIES, TYPENAMES, OPSYS_LIST
from src.model_registry import get_available_models
from src.inference import predict_price, format_inr
from src.ui import (
    apply_custom_css,
    render_page_header,
    render_metric_card,
    render_section_header,
    render_empty_state,
    lucide_icon
)

st.set_page_config(
    page_title="Price Predictor | Laptop Intelligence",
    layout="wide"
)

apply_custom_css()

render_page_header(
    title="Laptop Price Predictor",
    subtitle="Configure detailed hardware specifications to generate continuous market valuation estimates in Indian Rupees (₹).",
    icon_name="tag",
    eyebrow="Continuous Valuation"
)

available_models = get_available_models("regression")
if not available_models:
    st.error("No regression models found in `models/regression/`. Please run `python scripts/export_models.py`.")
    st.stop()

# Mode selection: Single Prediction vs Compare Two Configurations
mode = st.radio("Prediction Mode", ["Single Laptop Configuration", "Compare Two Configurations"], horizontal=True)

COMMON_RESOLUTIONS = [
    (1920, 1080, "Full HD (1920 x 1080)"),
    (1366, 768, "HD (1366 x 768)"),
    (2560, 1440, "QHD (2560 x 1440)"),
    (3840, 2160, "4K UHD (3840 x 2160)"),
    (2560, 1600, "Retina / WQXGA (2560 x 1600)"),
    (2880, 1800, "Retina 15\" (2880 x 1800)")
]

def render_laptop_form(prefix=""):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Identity & Platform**")
        company = st.selectbox(f"Brand / Manufacturer", COMPANIES, index=COMPANIES.index("Dell") if "Dell" in COMPANIES else 0, key=f"comp_{prefix}")
        typename = st.selectbox(f"Laptop Type", TYPENAMES, index=TYPENAMES.index("Notebook") if "Notebook" in TYPENAMES else 0, key=f"type_{prefix}")
        opsys = st.selectbox(f"Operating System", OPSYS_LIST, index=OPSYS_LIST.index("Windows 10") if "Windows 10" in OPSYS_LIST else 0, key=f"os_{prefix}")

    with c2:
        st.markdown("**Memory & Chassis**")
        ram = st.select_slider(f"RAM (GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8, key=f"ram_{prefix}")
        weight = st.number_input(f"Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1, key=f"wt_{prefix}")
        inches = st.number_input(f"Screen Size (inches)", min_value=10.0, max_value=18.5, value=15.6, step=0.1, key=f"inch_{prefix}")

    with c3:
        st.markdown("**Display & Storage**")
        res_idx = st.selectbox(
            f"Screen Resolution",
            range(len(COMMON_RESOLUTIONS)),
            format_func=lambda i: COMMON_RESOLUTIONS[i][2],
            index=0,
            key=f"res_{prefix}"
        )
        storage_gb = st.select_slider(f"Storage (GB)", options=[32, 64, 128, 256, 512, 1024, 2048], value=256, key=f"stor_{prefix}")
        
        st.write("")
        cb_col1, cb_col2 = st.columns(2)
        with cb_col1:
            has_ssd = st.checkbox(f"SSD Included", value=True, key=f"ssd_{prefix}")
            has_hdd = st.checkbox(f"HDD Included", value=False, key=f"hdd_{prefix}")
        with cb_col2:
            is_ips = st.checkbox(f"IPS Panel", value=False, key=f"ips_{prefix}")
            is_touch = st.checkbox(f"Touchscreen", value=False, key=f"touch_{prefix}")

    res_w, res_h, _ = COMMON_RESOLUTIONS[res_idx]

    return {
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

render_section_header("Inference Engine Selection", icon_name="cpu")
selected_model = st.selectbox("Select Fitted Model Pipeline", available_models, index=0)

if mode == "Single Laptop Configuration":
    render_section_header("Configuration Details", "Enter component specifications below", icon_name="laptop")
    
    with st.form("single_laptop_form", border=True):
        laptop_specs = render_laptop_form()
        submit_btn = st.form_submit_button("Predict Valuation", type="primary", use_container_width=True)

    if submit_btn:
        try:
            with st.spinner("Executing pipeline inference..."):
                result = predict_price(selected_model, laptop_specs)

            with st.container(border=True):
                st.markdown("### Valuation Estimate")
                res_c1, res_c2, res_c3 = st.columns(3)
                with res_c1:
                    render_metric_card(
                        label="Predicted Market Value",
                        value=result["formatted_price"],
                        subtext="Denominated in Indian Rupees",
                        icon_name="indian-rupee"
                    )
                with res_c2:
                    r2_str = f"{result['r2']:.4f}" if result['r2'] is not None else "N/A"
                    render_metric_card(
                        label="Model Test R² Score",
                        value=r2_str,
                        subtext=f"Pipeline: {selected_model}",
                        icon_name="target"
                    )
                with res_c3:
                    rmse_str = f"₹{result['rmse']:,.0f}" if result['rmse'] is not None else "N/A"
                    render_metric_card(
                        label="Test Root Mean Squared Error",
                        value=rmse_str,
                        subtext="Evaluation on holdout samples",
                        icon_name="trending-down"
                    )

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

else:
    render_section_header("Side-by-Side Configuration Comparison", "Evaluate price differentials between two alternative specifications", icon_name="scale")
    
    with st.form("comparison_form", border=False):
        comp_c1, comp_c2 = st.columns(2)

        with comp_c1:
            with st.container(border=True):
                st.subheader("Configuration A")
                specs_a = render_laptop_form(prefix="A")

        with comp_c2:
            with st.container(border=True):
                st.subheader("Configuration B")
                specs_b = render_laptop_form(prefix="B")

        compare_btn = st.form_submit_button("Compare Configurations", type="primary", use_container_width=True)

    if compare_btn:
        try:
            with st.spinner("Computing comparative valuations..."):
                res_a = predict_price(selected_model, specs_a)
                res_b = predict_price(selected_model, specs_b)

            price_a = res_a["predicted_price"]
            price_b = res_b["predicted_price"]
            delta = price_b - price_a

            with st.container(border=True):
                out_c1, out_c2, out_c3 = st.columns(3)
                with out_c1:
                    render_metric_card("Configuration A Price", res_a["formatted_price"], icon_name="laptop")
                with out_c2:
                    render_metric_card("Configuration B Price", res_b["formatted_price"], icon_name="laptop")
                with out_c3:
                    delta_str = f"{'+' if delta >= 0 else '-'}{format_inr(abs(delta))}"
                    render_metric_card(
                        label="Price Differential (B - A)",
                        value=delta_str,
                        delta=f"{abs(delta)/price_a*100:.1f}% vs Config A" if price_a > 0 else None,
                        delta_positive=(delta >= 0),
                        icon_name="scale"
                    )

        except Exception as e:
            st.error(f"Comparison Error: {str(e)}")
