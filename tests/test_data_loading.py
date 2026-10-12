import pandas as pd
from src.data_loader import (
    load_regression_data,
    load_classification_data,
    load_clustering_data,
    validate_dataset_schema,
    load_regression_metrics,
    load_classification_metrics
)

def test_load_regression_dataset():
    df = load_regression_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1243, 43)
    assert "price" in df.columns
    assert df["price"].isna().sum() == 0
    assert validate_dataset_schema(df, "regression")

def test_load_classification_dataset():
    df = load_classification_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1243, 43)
    assert "price_category" in df.columns
    assert "price" not in df.columns
    assert df["price_category"].isna().sum() == 0
    assert validate_dataset_schema(df, "classification")

def test_load_clustering_dataset_no_leakage():
    df = load_clustering_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1243, 42)
    assert "price" not in df.columns
    assert "price_category" not in df.columns
    assert validate_dataset_schema(df, "clustering")

def test_load_metrics():
    reg_metrics = load_regression_metrics()
    assert not reg_metrics.empty
    assert len(reg_metrics) == 10
    assert "R2" in reg_metrics.columns

    clf_metrics = load_classification_metrics()
    assert not clf_metrics.empty
    assert len(clf_metrics) == 10
    assert "Accuracy" in clf_metrics.columns
