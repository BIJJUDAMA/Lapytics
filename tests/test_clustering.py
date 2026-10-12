import json
import pandas as pd
from src.config import RESULTS_DIR, MODELS_DIR
from src.data_loader import load_clustering_data, load_clustering_metrics
from src.model_registry import load_model

def test_clustering_strictly_no_target_leakage():
    df = load_clustering_data()
    assert "price" not in df.columns
    assert "price_category" not in df.columns
    assert df.shape == (1243, 42)

def test_clustering_metrics_structure():
    metrics = load_clustering_metrics()
    assert "k_range" in metrics
    assert "kmeans_elbow" in metrics
    assert "silhouette_scores" in metrics
    assert "cluster_profiles" in metrics

    # Verify Silhouette scores are bounded between -1 and 1
    for s in metrics["silhouette_scores"]:
        assert -1.0 <= s <= 1.0

    # Verify cluster profiles contain descriptions
    for prof in metrics["cluster_profiles"]:
        assert "Summary" in prof
        assert len(prof["Summary"]) > 0

def test_clustering_models_loadable():
    scaler = load_model("clustering", "scaler")
    pca = load_model("clustering", "pca")
    kmeans_dict = load_model("clustering", "kmeans_dict")

    assert hasattr(scaler, "transform")
    assert hasattr(pca, "transform")
    assert 4 in kmeans_dict
    assert hasattr(kmeans_dict[4], "predict")
