# Machine Learning Capstone

## Source dataset and preprocessing

Raw input: [`laptop_price_uncleaned.csv`](laptop_price_uncleaned.csv). Original source: [Kaggle Uncleaned Laptop Price Dataset](https://www.kaggle.com/datasets/ehtishamsadiq/uncleaned-laptop-price-dataset).

[`pre_processing.ipynb`](pre_processing.ipynb) runs once on the raw dataset, then writes one CSV for each track. It does not create a train/test split and does not fit `StandardScaler`. Each track notebook performs its own split and scaling after loading its generated CSV.

### Common raw-data processing in `pre_processing.ipynb`

| Step                | Processing performed                                                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Column names        | Strips spaces, converts to lowercase, replaces spaces with`_`, removes special characters, drops `unnamed_0` when present                         |
| Numeric parsing     | Converts`price`, `ram`, `weight`, and `inches` from text to numeric values                                                                    |
| Missing values      | Fills missing`gpu` and `opsys` with `Unknown`; fills missing `weight` with its median; removes rows missing `price`, `ram`, or `inches` |
| Duplicates          | Removes duplicate rows                                                                                                                                |
| Category cleanup    | Standardizes`company`, `typename`, and `opsys` text values                                                                                      |
| Outliers            | Caps`price` and `weight` by the IQR bounds, `Q1 - 1.5 × IQR` and `Q3 + 1.5 × IQR`                                                           |
| Display features    | Extracts`res_width` and `res_height`; creates `is_ips` and `is_touchscreen`                                                                   |
| Storage features    | Extracts`storage_gb`; creates `has_ssd` and `has_hdd`                                                                                           |
| PPI feature         | Calculates`ppi = sqrt(res_width² + res_height²) / inches`                                                                                         |
| Encoding            | One-hot encodes`company`, `typename`, and `opsys`; converts boolean features to 0/1                                                             |
| Excluded raw fields | Excludes free-text`cpu`, `gpu`, `screenresolution`, and `memory` from the final model feature list                                            |

### Separate dataset creation per track

| Track          | Code path in`pre_processing.ipynb`                                                                          | Generated file              | Feature columns                          | Target handling                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------- | ---------------------------------------- | --------------------------------------------------------------------- |
| Regression     | `regression_df = df_encoded[encoded_feature_cols + ['price']]`                                              | `data/regression.csv`     | `encoded_feature_cols`, all 42 columns | Keeps continuous`price` as `y`                                    |
| Classification | `price_category = pd.qcut(df_encoded['price'], q=4, labels=['Budget', 'Mid-range', 'Premium', 'Flagship'])` | `data/classification.csv` | `encoded_feature_cols`, all 42 columns | Adds`price_category` as `y`; removes `price` to prevent leakage |
| Clustering     | `clustering_df = df_encoded[encoded_feature_cols]`                                                          | `data/clustering.csv`     | `encoded_feature_cols`, all 42 columns | Removes both`price` and `price_category`; no `y` exists         |

Run `pre_processing.ipynb` first. Then run the three track notebooks independently from their own generated CSV files.

## Dataset reference

| Track          | CSV                         | `X`                                 | `y`              |
| -------------- | --------------------------- | ------------------------------------- | ------------------ |
| Regression     | `data/regression.csv`     | Every column except`price`          | `price`          |
| Classification | `data/classification.csv` | Every column except`price_category` | `price_category` |
| Clustering     | `data/clustering.csv`     | Every column                          | None               |

Rows in each CSV: `1,243`

The supplied CSVs contain `42` feature columns. The assignment text refers to 29 features, but one-hot encoding expands the supplied data to 42 columns. Use all 42 columns together as the feature matrix. Do not train one model per column.

---

# Laptop Intelligence Dashboard & Deployment

A production-quality, interactive multi-page Streamlit web application providing continuous laptop price prediction (Regression), market tier classification (Classification), unsupervised architecture segmentation (Clustering), and an interactive model laboratory.

## Application Features

- **Laptop Price Predictor (Regression)**:
  - Interactive specification form supporting manufacturer, form factor, OS, RAM, screen size, resolution, PPI, and storage.
  - Generates estimated market valuations in Indian Rupees (₹).
  - Integrates 10 trained regression pipelines (Gradient Boosting Tuned: R² = 0.8121, Random Forest Tuned: R² = 0.8094, etc.).
  - Side-by-side comparison mode allowing price delta calculations between two configurations.
- **Laptop Price Classification (Classification)**:
  - Predicts 4 market price tiers (`Budget`, `Mid-range`, `Premium`, `Flagship`).
  - Displays predicted class confidence probabilities via interactive Plotly charts.
  - Benchmarks 10 classification algorithms with interactive confusion matrices and class distributions.
- **Laptop Clustering & Market Segmentation (Unsupervised)**:
  - Discovers latent laptop segments using K-Means and Agglomerative Hierarchical Clustering.
  - Zero target leakage: strictly operates on the 42 intrinsic hardware features without `price` or `price_category`.
  - 2D interactive PCA and t-SNE scatter plots with hover inspection.
  - Elbow curves, Silhouette scores, Davies-Bouldin, and Calinski-Harabasz metrics across cluster counts ($k=2..10$).
  - Automated descriptive summaries for discovered clusters.
- **Model Laboratory**:
  - Full leaderboard benchmarks across all three ML tracks.
  - Diagnostics: Actual vs. Predicted, Residual plots, and Feature Importance rankings.
  - Transparent provenance distinguishing notebook-origin metrics from standalone script extensions.
  - Read-only SHA-256 notebook integrity verification table.

## Repository Architecture

```text
ML_Capstone/
├── app.py                             # Main Streamlit dashboard landing page
├── pages/
│   ├── 1_Price_Predictor.py           # Regression price prediction interface
│   ├── 2_Price_Classification.py      # Price tier classification interface
│   ├── 3_Laptop_Clustering.py         # Unsupervised clustering and segmentation
│   └── 4_Model_Laboratory.py          # Model evaluation, diagnostics & leaderboards
├── src/
│   ├── __init__.py
│   ├── config.py                      # Feature schemas, bounds & hashes
│   ├── data_loader.py                 # Cached data loading & schema validation
│   ├── validation.py                  # User input validation & 42-feature vector builder
│   ├── model_registry.py              # Model discovery and lazy artifact loader
│   ├── inference.py                   # Prediction pipelines & INR formatting
│   └── visualization.py               # Plotly interactive charting utilities
├── scripts/
│   ├── export_models.py               # Standalone model training & serialization script
│   ├── generate_evaluation_results.py # Leaderboard and metrics evaluation script
│   └── prepare_clustering.py          # Clustering models & diagnostics preparation
├── models/
│   ├── regression/                    # Trained regression pipelines (.joblib)
│   ├── classification/                # Trained classification pipelines (.joblib)
│   └── clustering/                    # Scaler, PCA, and KMeans models (.joblib)
├── results/
│   ├── regression_metrics.csv         # Verified regression metrics
│   ├── classification_metrics.csv     # Verified classification metrics
│   ├── classification_confusion_matrices.json
│   ├── clustering_metrics.json        # Clustering diagnostics & profiles
│   ├── clustering_projections.csv     # 2D PCA & t-SNE coordinates
│   └── clustering_summary.csv
├── tests/
│   ├── test_audit_and_config.py
│   ├── test_data_loading.py
│   ├── test_validation.py
│   ├── test_inference.py
│   ├── test_clustering.py
│   └── test_app_structure.py
├── .streamlit/
│   └── config.toml                    # Streamlit dark theme configuration
├── conftest.py                        # Test runner path configuration
├── requirements.txt                   # Pinned application dependencies
├── .gitignore                         # Git exclusion rules
└── notebooks/                         # [READ-ONLY] Research notebooks
    ├── classification.ipynb
    ├── classification_10_algorithms.ipynb
    ├── clustering.ipynb
    ├── pre_processing.ipynb
    ├── Regression_algorithms_10.ipynb
    └── regression_first_5_algorithms.ipynb
```

## Local Setup and Execution

### 1. Prerequisites
- Python 3.10 to 3.12.

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/BIJJUDAMA/ML_Capstone.git
cd ML_Capstone
python -m pip install -r requirements.txt
```

### 3. Reproducing Models and Metrics
All model artifacts and metric tables can be reproduced independently of the Jupyter notebooks at any time:
```bash
# 1. Train and export regression and classification pipelines
python scripts/export_models.py

# 2. Generate evaluation metrics and leaderboards
python scripts/generate_evaluation_results.py

# 3. Prepare unsupervised clustering, PCA projections, and diagnostics
python scripts/prepare_clustering.py
```

### 4. Running the Dashboard
Launch the Streamlit web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

## Testing and Verification

Run the test suite using `pytest`:
```bash
python -m pytest -v
```
The test suite validates:
- Notebook SHA-256 hash preservation.
- Dataset schemas, shapes (1,243 rows, 42 features), and zero null values.
- Zero target leakage in clustering datasets and pipelines.
- User input boundary checks and feature vector construction.
- Model loading, inference, and INR currency formatting.
- Streamlit application and page rendering without runtime exceptions.

## Model Leaderboard Summary

### Regression (Test Set Metrics)
| Rank | Algorithm | R² Score | RMSE (₹) | MAE (₹) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Gradient Boosting Regressor (Tuned)** | **0.8121** | **₹14,869** | **₹11,026** |
| 2 | **Random Forest Regressor (Tuned)** | 0.8094 | ₹14,975 | ₹10,485 |
| 3 | Polynomial Regression (degree 2) | 0.7376 | ₹17,571 | ₹13,054 |
| 4 | K-Nearest Neighbors Regressor (Tuned) | 0.7118 | ₹18,417 | ₹12,834 |
| 5 | Decision Tree Regressor (Tuned) | 0.7010 | ₹18,757 | ₹13,154 |
| 6 | ElasticNet Regression | 0.6694 | ₹19,724 | ₹13,996 |
| 7 | Ridge Regression | 0.6652 | ₹19,848 | ₹13,895 |
| 8 | Lasso Regression | 0.6644 | ₹19,873 | ₹13,947 |
| 9 | Linear Regression | 0.6643 | ₹19,874 | ₹13,925 |
| 10 | Support Vector Regressor (Tuned) | 0.6643 | ₹19,876 | ₹13,796 |

### Classification (Test Set Metrics)
| Rank | Algorithm | Accuracy | Weighted F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Random Forest** | **70.68%** | **0.7080** | 0.7171 | 0.7068 |
| 2 | Bagging | 70.28% | 0.7009 | 0.7003 | 0.7028 |
| 3 | **SVC** | 69.08% | 0.6861 | 0.6920 | 0.6908 |
| 4 | Gradient Boosting | 67.87% | 0.6749 | 0.6776 | 0.6787 |
| 5 | Multi-layer Perceptron | 67.87% | 0.6767 | 0.6812 | 0.6787 |
| 6 | Logistic Regression | 65.06% | 0.6440 | 0.6411 | 0.6506 |
| 7 | KNN | 64.66% | 0.6401 | 0.6452 | 0.6466 |
| 8 | Decision Tree | 63.86% | 0.6155 | 0.6353 | 0.6386 |
| 9 | AdaBoost | 63.86% | 0.6302 | 0.6332 | 0.6386 |
| 10 | Gaussian Naive Bayes | 39.36% | 0.3216 | 0.5607 | 0.3936 |

### Clustering (Unsupervised Metrics & Segmentation)

Unsupervised market segmentation is performed strictly on the 42 intrinsic hardware features with zero target leakage (`price` and `price_category` excluded).

#### Algorithm & Linkage Comparison ($k=4$)
| Method / Linkage | Clusters ($k$) | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) |
| :--- | :---: | :---: | :---: | :---: |
| **K-Means (Partitioning)** | 4 | 0.1049 | 2.4182 | 91.26 |
| **Agglomerative (Ward Linkage)** | 4 | 0.1150 | 2.4929 | 82.14 |
| **Agglomerative (Complete Linkage)** | 4 | 0.6291 | 0.3291 | 42.59 |
| **Agglomerative (Average Linkage)** | 4 | 0.6588 | 0.2534 | 34.50 |

#### Discovered Hardware Archetype Profiles ($k=4$, K-Means)
| Cluster | Share | Count | Avg RAM | Avg Weight | Avg Storage | SSD % | Primary Archetype |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 9.1% | 113 | 12.6 GB | 1.54 kg | 495 GB | 88.5% | Premium Ultraportable / High-PPI (Touchscreen & Retina) |
| **1** | 30.7% | 381 | 6.1 GB | 2.21 kg | 934 GB | 12.9% | Budget Desktop Replacement (High Capacity HDD) |
| **2** | 44.7% | 556 | 7.5 GB | 1.74 kg | 253 GB | 89.4% | Mainstream Everyday Productivity (Fast SSD, Balanced) |
| **3** | 15.5% | 193 | 13.9 GB | 2.81 kg | 1,146 GB | 87.6% | High-Performance Workstation & Gaming (Heavy chassis) |

## Deployment Instructions

### Streamlit Community Cloud (Recommended)
1. Fork or push the repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app**.
4. Select repository: `BIJJUDAMA/ML_Capstone`, branch: `main`, main file path: `app.py`.
5. Under Advanced Settings, select Python 3.10, 3.11, or 3.12.
6. Click **Deploy**. The application will automatically install dependencies from `requirements.txt` and launch.

### Hugging Face Spaces (Alternative)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Streamlit** SDK.
3. Push the repository files to the Space git remote.

## Notebook Integrity Assurance

All original Jupyter research notebooks are located in `notebooks/` and remain strictly read-only artifacts. Their SHA-256 hashes are verified by automated tests:

| Notebook | SHA-256 Checksum | Status |
| :--- | :--- | :--- |
| `notebooks/classification.ipynb` | `767AB5C5D575B6837F1AD61ABC349B63610DA71F246D4A2BB873AF5399AC7C87` | Read-Only |
| `notebooks/classification_10_algorithms.ipynb` | `DA57C91281D871DE94A358235E901B8427C3F17922DA7B49C9D8E94D47FAD2BE` | Read-Only |
| `notebooks/clustering.ipynb` | `3A8FDDF90B5505B6008600B4A36D0FDBE172F53CD766798F5A454672B41337BC` | Read-Only |
| `notebooks/pre_processing.ipynb` | `F7822DE42521ECC3FA26213FCE98E9564D643D658139A8ACD6A22DB54376CA81` | Read-Only |
| `notebooks/Regression_algorithms_10.ipynb` | `2F12499D72172772ECB094B52C180885255BBB012CD7BBFC901C079FD97391B9` | Read-Only |
| `notebooks/regression_first_5_algorithms.ipynb` | `C652045DA75EA04DEEBB2C268C29ED025864588F04E3FF50ED73237D1173DF5A` | Read-Only |


