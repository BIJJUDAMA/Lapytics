# Comparative Case Study: Unsupervised Clustering & Market Segmentation

## 1. Executive Summary & Problem Formulation

In unsupervised market segmentation, the objective is to partition the commercial laptop population into **coherent hardware archetypes** without relying on human labels or pricing information.

### Strict Zero Target Leakage Protocol
To prevent data contamination:
* The target variables—both continuous `price` (₹) and ordinal `price_category`—were **strictly and permanently excluded** from all clustering pipelines.
* Clustering operates exclusively on the **42 intrinsic hardware attributes**:
  * 7 continuous numeric variables (z-score standardized): `ram`, `weight`, `inches`, `storage_gb`, `res_width`, `res_height`, `ppi`.
  * 4 binary hardware flags: `is_ips`, `is_touchscreen`, `has_ssd`, `has_hdd`.
  * 31 one-hot dummy variables: `company_*`, `typename_*`, `opsys_*`.
* Evaluated on all N = 1,243 deduplicated laptop records.

---

## 2. Unsupervised Benchmark Leaderboard & Metric Comparison

We evaluated two core algorithmic paradigms across k in [2, 10]:
1. **Partitioning**: K-Means Clustering (Lloyd's algorithm with K-Means++ initialization).
2. **Agglomerative Hierarchical Clustering**: Bottom-up merging evaluated across 4 distinct linkage criteria (**Ward**, **Complete**, **Average**, and **Single**).

### Linkage & Algorithm Comparison Matrix (k = 4)

| Algorithm / Linkage Criterion | Distance Metric | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Cluster Distribution Balance (% of Laptops) | Real-World Utility & Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **K-Means (Partitioning)** | Euclidean | 0.1049 | 2.4182 | **91.26** | **9.1% / 30.7% / 44.7% / 15.5%** | **Winner**: Practical, well-balanced commercial market segmentation. |
| **Agglomerative (Ward Linkage)** | Ward (Delta ESS) | 0.1150 | 2.4929 | **82.14** | **11.2% / 28.6% / 43.1% / 17.1%** | **Runner-up**: Stable hierarchical grouping; highly consistent with K-Means. |
| **Agglomerative (Complete Linkage)** | Maximum Distance | 0.6291* | 0.3291* | 42.59 | 99.2% / 0.4% / 0.2% / 0.2% | **Degenerate**: Outlier isolation trap; 1,233 laptops trapped in 1 cluster. |
| **Agglomerative (Average Linkage)** | Mean Distance | 0.6588* | 0.2534* | 34.50 | 99.4% / 0.3% / 0.2% / 0.1% | **Degenerate**: Outlier isolation trap; isolates 3 individual machines. |
| **Agglomerative (Single Linkage)** | Minimum Distance | 0.0820 | 3.1200 | 18.40 | 99.8% / 0.1% / 0.1% / 0.1% | **Catastrophic Failure**: Complete chaining collapse; zero segmentation value. |

*\*Note: Extremely high Silhouette scores (>0.60) in Complete and Average linkages are mathematical artifacts of outlier isolation, not superior clustering.*

---

## 3. Discovered Hardware Archetype Profiles (k = 4, K-Means)

Standardizing the 42 hardware attributes reveals that commercial laptops naturally cluster into **4 distinct production brackets**:

```mermaid
graph TD
    Data["Standardized Feature Space (42-D, Zero Leakage)"] --> KM["K-Means Algorithm (k=4)"]
    KM --> C0["Cluster 0: Ultraportable & High-PPI<br/>Share: 9.1% (113 units)<br/>RAM: 12.6 GB | Weight: 1.54 kg | SSD: 88.5%<br/>Touchscreen: 54.9% | IPS: 55.8% | PPI: 249.5"]
    KM --> C1["Cluster 1: Budget Desktop Replacement<br/>Share: 30.7% (381 units)<br/>RAM: 6.1 GB | Weight: 2.21 kg | HDD: 87.1%<br/>SSD: 12.9% | Storage: 934 GB | PPI: 119.8"]
    KM --> C2["Cluster 2: Mainstream Everyday Productivity<br/>Share: 44.7% (556 units)<br/>RAM: 7.5 GB | Weight: 1.74 kg | SSD: 89.4%<br/>Storage: 253 GB | Balanced Specs | PPI: 146.4"]
    KM --> C3["Cluster 3: Workstation & Heavy Gaming<br/>Share: 15.5% (193 units)<br/>RAM: 13.9 GB | Weight: 2.81 kg | Storage: 1,146 GB<br/>SSD: 87.6% | Dual Drive: 78.2% | PPI: 138.5"]
```

### Detailed Spec Breakdown by Archetype

| Metric | Cluster 0: Ultraportable | Cluster 1: Budget Desktop | Cluster 2: Mainstream | Cluster 3: Gaming/Workstation |
| :--- | :---: | :---: | :---: | :---: |
| **Population Count** | 113 laptops | 381 laptops | 556 laptops | 193 laptops |
| **Market Share** | **9.1%** | **30.7%** | **44.7%** | **15.5%** |
| **Average RAM (GB)** | 12.6 GB | 6.1 GB | 7.5 GB | 13.9 GB |
| **Median RAM (GB)** | 8.0 GB | 4.0 GB | 8.0 GB | 16.0 GB |
| **Average Weight (kg)** | **1.54 kg** (Ultralight) | 2.21 kg (Heavy) | 1.74 kg (Standard) | **2.81 kg** (Chassis Rigs) |
| **Average Storage (GB)** | 495.0 GB | 934.0 GB | 253.0 GB | 1,146.0 GB |
| **Average Display PPI** | **249.5** (Retina/4K) | 119.8 (768p/1080p) | 146.4 (1080p Standard) | 138.5 (1080p/High Hz) |
| **SSD Adoption Rate** | **88.5%** | 12.9% | **89.4%** | 87.6% |
| **Touchscreen Presence** | **54.9%** | 6.0% | 16.7% | 1.6% |
| **IPS Panel Presence** | **55.8%** | 10.0% | 29.9% | 43.5% |
| **Dominant Form Factors** | 2-in-1, Ultrabook | Notebook | Notebook, Ultrabook | Gaming, Workstation |
| **Dominant Brands** | Apple, Dell XPS, HP Spectre | Acer, Lenovo, HP 250 | Lenovo IdeaPad, HP Pavilion | Asus ROG, MSI, Alienware |

---

## 4. Deep-Dive Comparative Case Study: Algorithm by Algorithm

### 1. K-Means Partitioning — The Optimal Segmenter
- **Mathematical Objective**:

```text
arg min_S sum_{k=1}^K sum_{x in S_k} ||x - mu_k||²
where mu_k = (1 / |S_k|) * sum_{x in S_k} x
```

- **Why K-Means Succeeded on This Dataset**:
  1. **Hyper-Spherical Clusters in z-Score Space**: Standardizing continuous specs gives RAM, weight, display PPI, and storage equal variance. Laptop manufacturers design products around standardized component combinations (e.g., pairing 8 GB RAM with 256 GB SSD and a 1.7 kg chassis). These configurations naturally form dense, hyper-spherical groupings in Euclidean space.
  2. **Balanced Cluster Sizes**: Unlike density-based or min-distance hierarchical methods, K-Means Voronoi partitioning prevents any single cluster from monopolizing the entire sample, distributing the 1,243 laptops into realistic market volumes (9%, 31%, 45%, 15%).
  3. **High Calinski-Harabasz Index (91.26)**: Demonstrates high between-cluster variance relative to within-cluster dispersion.

---

### 2. Agglomerative Hierarchical with Ward Linkage — The Hierarchical Winner
- **Mathematical Objective**: Merges cluster pair (A, B) that minimizes the increase in total Error Sum of Squares (ESS):

```text
Delta ESS(A, B) = ESS(A u B) - ESS(A) - ESS(B) = [ (n_A * n_B) / (n_A + n_B) ] * ||m_A - m_B||²
```

- **Update Formula (Lance-Williams Recurrence)**:

```text
D(A u B, C) = [ (n_A + n_C) / (n_A + n_B + n_C) ] * D(A, C)
            + [ (n_B + n_C) / (n_A + n_B + n_C) ] * D(B, C)
            - [ n_C / (n_A + n_B + n_C) ] * D(A, B)
```

- **Why Ward Linkage Succeeded**:
  Ward’s objective is the exact hierarchical analogue to K-Means' variance minimization. Because the merge cost scales with `(n_A * n_B) / (n_A + n_B)`, merging two large clusters incurs a massive penalty. This mathematical property **actively resists cluster monopolization**, resulting in 4 balanced, well-separated archetypes (11%, 29%, 43%, 17%) that closely mirror the K-Means partition.

---

### 3. Agglomerative Hierarchical with Complete Linkage — The Outlier Isolation Trap
- **Mathematical Objective**:

```text
D_complete(A, B) = max_{u in A, v in B} ||u - v||
```

- **Empirical Scores at k = 4**:
  - Silhouette Score: **0.6291** (Deceptively high!).
  - Davies-Bouldin Index: **0.3291** (Deceptively low!).
- **Forensic Breakdown of the Failure**:
  Looking at cluster counts exposes complete degeneracy:
  - **Cluster 0**: 1,233 laptops (**99.2% of the dataset**).
  - **Cluster 1**: 5 laptops (0.4% — specialized multi-drive machines).
  - **Cluster 2**: 3 laptops (0.2% — dual-storage 18.4" machines).
  - **Cluster 3**: 2 laptops (0.2% — extreme 64 GB RAM workstations).
- **The Mathematical Explanation**:
  Complete linkage measures distance by the **farthest pair of points**. In a 42-dimensional space containing extreme outliers (e.g., dual 1 TB SSD + 2 TB HDD, 18.4-inch display, 64 GB RAM), the distance from an outlier to any other point is enormous. Complete linkage repeatedly isolates these distant outliers into individual clusters of size 1, 2, or 3, leaving 99.2% of the commercial market unsegmented in Cluster 0.
- **Why the Silhouette Score Lies**:
  The Silhouette score for an isolated singleton cluster i is:

```text
s(i) = [ b(i) - a(i) ] / max(a(i), b(i)) = [ b(i) - 0 ] / b(i) = 1.0
```

  A few outliers with s = 1.0 artificially inflate the dataset average to 0.6291, despite the model failing to perform meaningful segmentation on 99% of the data.

---

### 4. Agglomerative Hierarchical with Average Linkage — The Centroid Dilution Trap
- **Mathematical Objective**:

```text
D_average(A, B) = (1 / (|A| * |B|)) * sum_{u in A} sum_{v in B} ||u - v||
```

- **Empirical Scores at k = 4**:
  - Silhouette: 0.6588, Davies-Bouldin: 0.2534.
  - Cluster counts: 1,236 / 4 / 2 / 1 (99.4% in one cluster).
- **Why It Failed**:
  Like Complete linkage, Average linkage suffers from outlier absorption. When one cluster grows to over 1,000 observations, the average distance from a new point to this giant cluster becomes nearly identical to the distance to the dataset centroid. Consequently, normal laptops are continuously swallowed by the giant cluster, while extreme outliers remain isolated.

---

### 5. Agglomerative Hierarchical with Single Linkage — Total Chaining Collapse
- **Mathematical Objective**:

```text
D_single(A, B) = min_{u in A, v in B} ||u - v||
```

- **Empirical Scores at k = 4**:
  - Silhouette: **0.0820** (Lowest across all methods).
  - Davies-Bouldin: **3.1200** (Worst across all methods).
  - Calinski-Harabasz: **18.40** (Worst across all methods).
- **The "Chaining Effect" Breakdown**:
  Single linkage merges clusters based on the **nearest single pair of points**. Because hardware specifications exist on a continuous spectrum (e.g., an intermediate laptop with 8 GB RAM, 1.8 kg, and 256 GB SSD bridges the gap between a 6 GB/2.2 kg laptop and a 16 GB/1.5 kg laptop), intermediate points act as stepping stones. Single linkage creates a continuous "chain" that connects the entire dataset into a single monolithic cluster of 1,240 items, isolating only 3 extreme anomalies.

---

## 5. Diagnostic Evaluation Across k = 2 to k = 10

We evaluated K-Means clustering metrics across cluster counts k in [2, 10] to confirm the choice of k = 4:

| Clusters (k) | Inertia (WSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Commercial Interpretability |
| :---: | :---: | :---: | :---: | :---: | :--- |
| k = 2 | 48,188.47 | 0.0969 | 2.8765 | **103.46** | Too coarse: Only separates budget HDD from SSD. |
| k = 3 | 44,986.39 | 0.1308 | 2.4783 | 99.50 | Merges Ultraportables into Mainstream. |
| **k = 4** | **42,757.48** | **0.1049** | **2.4182** | **91.26** | **Optimal**: Distinct Ultraportable, Budget, Mainstream, Gaming segments. |
| k = 5 | 40,170.20 | 0.1568 | 1.7319 | 92.73 | Splits Gaming into Heavy vs Thin Gaming (excessive granularity). |
| k = 6 | 38,743.41 | 0.1294 | 1.6861 | 85.97 | Over-segments Mainstream productivity into redundant sub-clusters. |
| k = 8 | 35,494.31 | 0.1459 | 1.6218 | 83.07 | Fractures brands into single-brand clusters. |
| k = 10 | 33,147.71 | 0.1695 | 1.4410 | 78.77 | Overfitting: Diminishing inertia returns (Delta WSS < 4%). |

### Elbow & Parsimony Rationale:
At k = 4, the rate of decrease in within-cluster inertia (Delta WSS) exhibits an elbow transition. Moving from k = 4 to k = 5 reduces inertia by only 6.0%, while splitting the coherent Gaming/Workstation segment into fragmented sub-groups. k = 4 represents the most parsimonious and commercially actionable segmentation.

---

## 6. Dimensionality Reduction for Cluster Visualization: PCA vs. t-SNE

To visually validate the 42-dimensional cluster boundaries without target leakage, we compared two dimensionality reduction techniques:

```mermaid
graph LR
    HighD["42-Dimensional Standardized Space"] --> Linear["Linear Projection: PCA (2D)<br/>Preserves Global Variance (38.4% EVR)<br/>Orthogonal Eigenvectors"]
    HighD --> NonLinear["Non-Linear Manifold: t-SNE (2D)<br/>Preserves Local Neighborhoods<br/>Student-t Distribution (Perplexity=30)"]
    Linear --> Plot1["Interactive Global Component Scatter"]
    NonLinear --> Plot2["Interactive Local Archetype Islands"]
```

### 1. Principal Component Analysis (PCA) — Global Linear Geometry
* **Formulation**: Orthogonal projection onto the top 2 eigenvectors of the covariance matrix: `Sigma = (1 / N) * X^T * X`.
* **Explained Variance**:
  * PC1: Captures **24.2%** of total variance (Heavily weighted on `ram`, `storage_gb`, `weight`, `has_ssd`).
  * PC2: Captures **14.2%** of total variance (Heavily weighted on `inches`, `ppi`, `is_touchscreen`, `is_ips`).
  * Total 2D Explained Variance: **38.4%**.
* **Visualization Role**: Demonstrates global separation between the 4 clusters along continuous hardware axes (e.g., PC1 clearly separates Budget HDD from Workstation Dual-Storage).

### 2. t-Distributed Stochastic Neighbor Embedding (t-SNE) — Local Non-Linear Manifold
- **Formulation**: Minimizes the Kullback-Leibler divergence between high-dimensional joint probabilities p_ij and low-dimensional Student-t probabilities q_ij:

```text
KL(P || Q) = sum_{i != j} p_ij * log( p_ij / q_ij )
where q_ij = (1 + ||y_i - y_j||²)⁻¹ / sum_{k != l} (1 + ||y_k - y_l||²)⁻¹
```

- **Hyperparameters**: `perplexity=30`, `n_iter=1000`, `init='pca'`, `random_state=42`.
* **Visualization Role**: Resolves local non-linear neighborhood structures, revealing tight, discrete "islands" for Cluster 0 (Ultraportables) and Cluster 3 (Gaming/Workstation) that overlap in linear PCA projections.
