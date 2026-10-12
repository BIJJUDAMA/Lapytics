import numpy as np
import pandas as pd

from src.config import PRICE_CATEGORIES
from src.data_loader import load_classification_metrics, load_regression_metrics
from src.model_registry import load_model
from src.validation import build_feature_vector, validate_laptop_input

REGRESSION_LIMITATIONS = (
    "Predictions are statistical estimates generated from the 1,243 laptop capstone dataset in Indian Rupees (INR). "
    "In the preprocessing pipeline, prices were capped using IQR boundaries (₹9,270 to ₹1,50,358). "
    "Predictions do not account for daily currency fluctuations, regional retailer discounts, or newly released chip architectures."
)

CLASSIFICATION_LIMITATIONS = (
    "Price category classification is based on historical quartile boundaries of the cleaned dataset: "
    "Budget (≤ ₹32,650), Mid-range (₹32,650–₹52,694), Premium (₹52,694–₹79,734), and Flagship (> ₹79,734). "
    "Thresholds are derived from the capstone sample distribution."
)

def format_inr(amount: float) -> str:
    """Formats a float as an Indian Rupee currency string."""
    amt_int = int(round(amount))
    s = str(amt_int)
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    # In Indian numbering, thousands are grouped in pairs of two
    groups = []
    while len(remaining) > 2:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.insert(0, remaining)
    formatted = ",".join(groups) + "," + last_three
    return f"₹{formatted}"


def predict_price(model_name: str, input_dict: dict) -> dict:
    """
    Predicts the continuous price of a laptop given user specs and chosen regression model.
    """
    is_valid, errors = validate_laptop_input(input_dict)
    if not is_valid:
        raise ValueError("; ".join(errors))

    model = load_model("regression", model_name)
    features_df = build_feature_vector(input_dict)

    pred_raw = float(model.predict(features_df)[0])
    # Ensure reasonable non-negative prediction
    pred_price = max(0.0, pred_raw)

    # Lookup evaluation metrics for this model
    reg_metrics = load_regression_metrics()
    model_row = reg_metrics[reg_metrics["Algorithm"] == model_name]
    r2_val = float(model_row["R2"].values[0]) if not model_row.empty else None
    rmse_val = float(model_row["RMSE"].values[0]) if not model_row.empty else None
    mae_val = float(model_row["MAE"].values[0]) if not model_row.empty else None

    return {
        "model_name": model_name,
        "predicted_price": pred_price,
        "formatted_price": format_inr(pred_price),
        "r2": r2_val,
        "rmse": rmse_val,
        "mae": mae_val,
        "limitations": REGRESSION_LIMITATIONS
    }


def predict_price_category(model_name: str, input_dict: dict) -> dict:
    """
    Predicts the price tier (Budget, Mid-range, Premium, Flagship) for a laptop configuration.
    """
    is_valid, errors = validate_laptop_input(input_dict)
    if not is_valid:
        raise ValueError("; ".join(errors))

    model = load_model("classification", model_name)
    features_df = build_feature_vector(input_dict)

    pred_category = str(model.predict(features_df)[0])

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(features_df)[0]
            classes = model.classes_
            probabilities = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)}
        except Exception:
            probabilities = None

    # Lookup evaluation metrics for this classifier
    clf_metrics = load_classification_metrics()
    model_row = clf_metrics[clf_metrics["Algorithm"] == model_name]
    acc_val = float(model_row["Accuracy"].values[0]) if not model_row.empty else None
    f1_val = float(model_row["Weighted_F1"].values[0]) if not model_row.empty else None

    return {
        "model_name": model_name,
        "predicted_category": pred_category,
        "probabilities": probabilities,
        "accuracy": acc_val,
        "weighted_f1": f1_val,
        "limitations": CLASSIFICATION_LIMITATIONS
    }
