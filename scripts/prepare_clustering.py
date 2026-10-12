"""
Clustering and Segmentation Preparation Script
Performs unsupervised learning on data/clustering.csv with ZERO target leakage.
Computes K-Means, Agglomerative clustering, PCA, t-SNE, diagnostics, and cluster profiles.
"""

import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

def prepare_clustering():
    print("Preparing clustering analysis and artifacts...")
    df = pd.read_csv(DATA_DIR / "clustering.csv")

    # Strict target leakage check
    if "price" in df.columns or "price_category" in df.columns:
        raise ValueError("CRITICAL: Target column found in clustering dataset!")

    print(f"Loaded clustering features: {df.shape} (42 features, zero target leakage)")

    # 1. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    clust_dir = MODELS_DIR / "clustering"
    clust_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, clust_dir / "scaler.joblib")

    # 2. KMeans diagnostics for k=2..10
    k_range = list(range(2, 11))
    kmeans_elbow = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    kmeans_models = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)

        kmeans_elbow.append(round(float(km.inertia_), 2))
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(round(float(sil), 4))
        db = davies_bouldin_score(X_scaled, labels)
        davies_bouldin_scores.append(round(float(db), 4))
        ch = calinski_harabasz_score(X_scaled, labels)
        calinski_harabasz_scores.append(round(float(ch), 2))

        if k <= 6:
            kmeans_models[k] = km

    joblib.dump(kmeans_models, clust_dir / "kmeans_models.joblib")

    # 3. Agglomerative Linkage Comparison for k=2..6 matching clustering.ipynb LINKAGES
    linkage_comparison = []
    for linkage in ["ward", "complete", "average", "single"]:
        for k in range(2, 7):
            agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = agg.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            linkage_comparison.append({
                "Linkage": linkage,
                "k": k,
                "Silhouette": round(float(sil), 4),
                "Davies_Bouldin": round(float(db), 4),
                "Calinski_Harabasz": round(float(ch), 2)
            })

    # 4. Dimensionality Reduction (PCA and t-SNE)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, clust_dir / "pca_model.joblib")

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    # Base projection DataFrame
    proj_df = pd.DataFrame({
        "PCA_1": np.round(X_pca[:, 0], 4),
        "PCA_2": np.round(X_pca[:, 1], 4),
        "tSNE_1": np.round(X_tsne[:, 0], 4),
        "tSNE_2": np.round(X_tsne[:, 1], 4),
        "ram": df["ram"],
        "weight": df["weight"],
        "inches": df["inches"],
        "storage_gb": df["storage_gb"],
        "ppi": np.round(df["ppi"], 1)
    })

    # Attach cluster assignments for k=2..6 for KMeans and Agglomerative (Ward)
    for k in range(2, 7):
        proj_df[f"kmeans_cluster_{k}"] = kmeans_models[k].labels_
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        proj_df[f"agg_cluster_{k}"] = agg.fit_predict(X_scaled)

    proj_path = RESULTS_DIR / "clustering_projections.csv"
    proj_df.to_csv(proj_path, index=False)
    print(f"Saved projections to {proj_path}")

    # 5. Cluster Profiles generation (for default k=4)
    km4_labels = kmeans_models[4].labels_
    df_with_cluster = df.copy()
    df_with_cluster["cluster"] = km4_labels

    cluster_profiles = []
    total_laptops = len(df)

    for c in range(4):
        c_df = df_with_cluster[df_with_cluster["cluster"] == c]
        count = len(c_df)
        pct = round(100.0 * count / total_laptops, 1)

        # Dominant specs
        avg_ram = round(float(c_df["ram"].mean()), 1)
        med_ram = round(float(c_df["ram"].median()), 1)
        avg_weight = round(float(c_df["weight"].mean()), 2)
        avg_storage = round(float(c_df["storage_gb"].mean()), 0)
        avg_ppi = round(float(c_df["ppi"].mean()), 1)
        ssd_ratio = round(float(c_df["has_ssd"].mean() * 100), 1)
        touch_ratio = round(float(c_df["is_touchscreen"].mean() * 100), 1)
        ips_ratio = round(float(c_df["is_ips"].mean() * 100), 1)

        # Generate automatic descriptive summary
        desc = []
        if avg_ram >= 16 or avg_weight >= 2.5:
            desc.append("High-Performance Workstations & Gaming (Large RAM, heavier chassis)")
        elif avg_weight <= 1.5 and ssd_ratio >= 80:
            desc.append("Ultraportable & Premium Laptops (Lightweight, SSD, High PPI)")
        elif avg_ram <= 6 and ssd_ratio < 50:
            desc.append("Entry-Level & Budget Laptops (Standard HDD/eMMC, lower RAM)")
        else:
            desc.append("Mainstream Productivity & All-Rounder Notebooks (Balanced specs)")

        cluster_profiles.append({
            "Cluster": c,
            "Count": count,
            "Percentage": pct,
            "Avg_RAM_GB": avg_ram,
            "Median_RAM_GB": med_ram,
            "Avg_Weight_kg": avg_weight,
            "Avg_Storage_GB": avg_storage,
            "Avg_PPI": avg_ppi,
            "SSD_Percentage": ssd_ratio,
            "Touchscreen_Percentage": touch_ratio,
            "IPS_Percentage": ips_ratio,
            "Summary": desc[0]
        })

    # Save metrics JSON
    metrics_data = {
        "k_range": k_range,
        "kmeans_elbow": kmeans_elbow,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
        "calinski_harabasz_scores": calinski_harabasz_scores,
        "linkage_comparison": linkage_comparison,
        "pca_explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "cluster_profiles": cluster_profiles
    }

    metrics_path = RESULTS_DIR / "clustering_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved clustering metrics and profiles to {metrics_path}")

    # Summary table
    prof_df = pd.DataFrame(cluster_profiles)
    prof_path = RESULTS_DIR / "clustering_summary.csv"
    prof_df.to_csv(prof_path, index=False)
    print(prof_df.to_string(index=False))

if __name__ == "__main__":
    prepare_clustering()
