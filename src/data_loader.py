import json
from pathlib import Path
import streamlit as st
import pandas as pd

from src.config import (
    DATA_DIR,
    RESULTS_DIR,
    ALL_FEATURE_COLUMNS,
    PRICE_CATEGORIES
)

@st.cache_data
def load_regression_data() -> pd.DataFrame:
    path = DATA_DIR / "regression.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "price" not in df.columns:
        raise ValueError(f"'price' target column missing from {path}")
    return df

@st.cache_data
def load_classification_data() -> pd.DataFrame:
    path = DATA_DIR / "classification.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "price_category" not in df.columns:
        raise ValueError(f"'price_category' target column missing from {path}")
    return df

@st.cache_data
def load_clustering_data() -> pd.DataFrame:
    path = DATA_DIR / "clustering.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "price" in df.columns or "price_category" in df.columns:
        raise ValueError(f"Target leakage detected in {path}! Targets must not exist in clustering.")
    return df

def validate_dataset_schema(df: pd.DataFrame, task: str) -> bool:
    if task == "regression":
        expected_cols = set(ALL_FEATURE_COLUMNS + ["price"])
        return set(df.columns) == expected_cols and len(df) == 1243
    elif task == "classification":
        expected_cols = set(ALL_FEATURE_COLUMNS + ["price_category"])
        return set(df.columns) == expected_cols and len(df) == 1243
    elif task == "clustering":
        expected_cols = set(ALL_FEATURE_COLUMNS)
        return set(df.columns) == expected_cols and len(df) == 1243
    return False

@st.cache_data
def load_regression_metrics() -> pd.DataFrame:
    path = RESULTS_DIR / "regression_metrics.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Rank", "Algorithm", "R2", "RMSE", "MAE", "CV_R2_Mean", "CV_R2_Std"])
    return pd.read_csv(path)

@st.cache_data
def load_classification_metrics() -> pd.DataFrame:
    path = RESULTS_DIR / "classification_metrics.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Rank", "Algorithm", "Accuracy", "Weighted_F1", "Precision", "Recall", "ROC_AUC"])
    return pd.read_csv(path)

@st.cache_data
def load_clustering_metrics() -> dict:
    path = RESULTS_DIR / "clustering_metrics.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_clustering_projections() -> pd.DataFrame:
    path = RESULTS_DIR / "clustering_projections.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
