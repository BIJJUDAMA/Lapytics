# Laptop Price Intelligence Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade interactive Streamlit web dashboard for Laptop Price Prediction and Analysis (Regression, Classification, Clustering, and Model Laboratory) with reproducible scripts, tests, verified metrics, and deployment readiness, while leaving existing Jupyter notebooks completely untouched.

**Architecture:** A modular Python architecture separating core data contracts, schema validation, model registry, inference pipeline, visualization helpers, and data preparation scripts from the multi-page Streamlit UI (`app.py` and `pages/1..4`). Supervised and unsupervised models are trained and serialized using standalone reproducible scripts that mirror and extend existing notebook experiments.

**Tech Stack:** Python 3.12, Streamlit 1.65+, scikit-learn 1.8+, Pandas 2.2+, NumPy 2.3+, Plotly 6.7+, Joblib 1.5+, Pytest 9.0+.

**Spec:** User request and capstone requirements in workspace `C:\My-Files\Github\ML_Capstone`.

## Global Constraints

- **CRITICAL**: DO NOT MODIFY, MOVE, RENAME, RE-EXECUTE, OR OVERWRITE EXISTING NOTEBOOKS (`classification.ipynb`, `pre_processing.ipynb`, `Regression_algorithms_10.ipynb`, `regression_first_5_algorithms.ipynb`). Their SHA-256 hashes must remain strictly identical before and after.
- **NO TARGET LEAKAGE**: `price` and `price_category` must never be present in clustering inputs, clustering datasets, or clustering models.
- **NO FABRICATED METRICS**: Only report metrics derived from verified notebook outputs or freshly executed, reproducible training and evaluation scripts. Clearly distinguish original notebook results from script-extended results.
- **NO FITTING ON PREDICTION INPUT**: Never fit scalers, encoders, or imputers on live user input. Preprocessing must use pre-fitted pipelines.
- **REPRODUCIBILITY**: `random_state=42` must be used across all splits, models, and clustering algorithms to match existing notebooks.

---

### Task 1: Audit Verification, Initial Notebook Hashes, and Project Configuration Scaffolding

**Files:**
- Create: `.streamlit/config.toml`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Test: `tests/test_audit_and_config.py`

**Interfaces:**
- Consumes: Nothing
- Produces:
  - `src.config.BASE_DIR`: Path
  - `src.config.DATA_DIR`: Path
  - `src.config.MODELS_DIR`: Path
  - `src.config.RESULTS_DIR`: Path
  - `src.config.ALL_FEATURE_COLUMNS`: list[str] (42 columns)
  - `src.config.NUMERIC_FEATURES`: list[str]
  - `src.config.BOOLEAN_FEATURES`: list[str]
  - `src.config.CATEGORICAL_OPTIONS`: dict[str, list[str]]
  - `src.config.NOTEBOOK_HASHES`: dict[str, str]

- [ ] **Step 1: Write the failing test**

Create `tests/test_audit_and_config.py`:
```python
import hashlib
from pathlib import Path
from src.config import NOTEBOOK_HASHES, ALL_FEATURE_COLUMNS, BASE_DIR

def test_notebook_hashes_unchanged():
    for nb_name, expected_hash in NOTEBOOK_HASHES.items():
        nb_path = BASE_DIR / nb_name
        assert nb_path.exists(), f"Notebook {nb_name} does not exist"
        content = nb_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest().upper()
        assert actual_hash == expected_hash, f"Hash mismatch for {nb_name}: {actual_hash} != {expected_hash}"

def test_feature_columns_count():
    assert len(ALL_FEATURE_COLUMNS) == 42
    assert "inches" in ALL_FEATURE_COLUMNS
    assert "ppi" in ALL_FEATURE_COLUMNS
    assert "price" not in ALL_FEATURE_COLUMNS
    assert "price_category" not in ALL_FEATURE_COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit_and_config.py`
Expected: FAIL (ModuleNotFoundError: No module named 'src')

- [ ] **Step 3: Implement minimal configuration and scaffolding**

Create `src/__init__.py`.

Create `src/config.py` with exact paths, 42 feature columns, category options, bounds, and recorded SHA-256 hashes:
- `classification.ipynb`: `767AB5C5D575B6837F1AD61ABC349B63610DA71F246D4A2BB873AF5399AC7C87`
- `pre_processing.ipynb`: `F7822DE42521ECC3FA26213FCE98E9564D643D658139A8ACD6A22DB54376CA81`
- `Regression_algorithms_10.ipynb`: `2F12499D72172772ECB094B52C180885255BBB012CD7BBFC901C079FD97391B9`
- `regression_first_5_algorithms.ipynb`: `C652045DA75EA04DEEBB2C268C29ED025864588F04E3FF50ED73237D1173DF5A`

Create `.streamlit/config.toml` configuring the dark theme:
```toml
[theme]
base = "dark"
primaryColor = "#3B82F6"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#F8FAFC"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = true
```

Create `.gitignore` ignoring caches, virtual environments, `.pytest_cache`, and temporary files.

Create `requirements.txt` with pinned, compatible versions:
```text
streamlit>=1.40.0,<=1.65.0
scikit-learn>=1.5.0,<=1.8.0
pandas>=2.2.0,<=2.3.0
numpy>=1.26.0,<=2.4.0
plotly>=5.20.0,<=6.7.0
joblib>=1.4.0,<=1.6.0
scipy>=1.13.0,<=1.17.0
pytest>=8.0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audit_and_config.py`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit scaffolding**

```bash
git add src/ tests/ .streamlit/ .gitignore requirements.txt
git commit -m "feat: setup project scaffolding, config, and notebook audit hashes"
```

---

### Task 2: Data Loader and Dataset Integrity Layer

**Files:**
- Create: `src/data_loader.py`
- Test: `tests/test_data_loading.py`

**Interfaces:**
- Consumes: `src.config` (paths, column definitions)
- Produces:
  - `load_regression_data()` -> `pd.DataFrame` (1243 rows, 43 cols including `price`)
  - `load_classification_data()` -> `pd.DataFrame` (1243 rows, 43 cols including `price_category`)
  - `load_clustering_data()` -> `pd.DataFrame` (1243 rows, 42 cols, zero targets)
  - `load_regression_metrics()` -> `pd.DataFrame`
  - `load_classification_metrics()` -> `pd.DataFrame`
  - `load_clustering_metrics()` -> `dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_loading.py`:
```python
import pandas as pd
from src.data_loader import (
    load_regression_data,
    load_classification_data,
    load_clustering_data,
    validate_dataset_schema
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_loading.py`
Expected: FAIL (ModuleNotFoundError: No module named 'src.data_loader')

- [ ] **Step 3: Implement data loader**

Create `src/data_loader.py`:
- Implement `load_regression_data`, `load_classification_data`, `load_clustering_data` with caching (`@functools.lru_cache` or standard loading).
- Implement `validate_dataset_schema` verifying column lists and row count (1243).
- Implement `load_regression_metrics`, `load_classification_metrics`, and `load_clustering_metrics` with safe fallbacks if files have not been generated yet.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_loading.py`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit data loader**

```bash
git add src/data_loader.py tests/test_data_loading.py
git commit -m "feat: implement data loading and schema validation"
```

---

### Task 3: Input Validation and Feature Vector Transformation

**Files:**
- Create: `src/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `src.config`
- Produces:
  - `validate_laptop_input(input_dict: dict) -> tuple[bool, list[str]]`
  - `build_feature_vector(input_dict: dict) -> pd.DataFrame` (1 row, 42 columns matching `ALL_FEATURE_COLUMNS`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_validation.py`:
```python
import pandas as pd
from src.validation import validate_laptop_input, build_feature_vector
from src.config import ALL_FEATURE_COLUMNS

def test_valid_laptop_input():
    sample = {
        "company": "Dell",
        "typename": "Notebook",
        "opsys": "Windows 10",
        "ram": 16.0,
        "weight": 2.1,
        "inches": 15.6,
        "res_width": 1920,
        "res_height": 1080,
        "storage_gb": 512.0,
        "has_ssd": 1,
        "has_hdd": 0,
        "is_ips": 1,
        "is_touchscreen": 0
    }
    is_valid, errors = validate_laptop_input(sample)
    assert is_valid
    assert len(errors) == 0

    vec = build_feature_vector(sample)
    assert isinstance(vec, pd.DataFrame)
    assert vec.shape == (1, 42)
    assert list(vec.columns) == ALL_FEATURE_COLUMNS
    assert vec.loc[0, "company_Dell"] == 1
    assert vec.loc[0, "company_Hp"] == 0
    assert vec.loc[0, "ram"] == 16.0
    expected_ppi = (1920**2 + 1080**2)**0.5 / 15.6
    assert abs(vec.loc[0, "ppi"] - expected_ppi) < 1e-4

def test_invalid_laptop_input_bounds():
    sample = {
        "company": "InvalidBrand",
        "typename": "Notebook",
        "opsys": "Windows 10",
        "ram": -5.0,  # invalid
        "weight": 2.1,
        "inches": 15.6,
        "res_width": 1920,
        "res_height": 1080,
        "storage_gb": 512.0,
        "has_ssd": 1,
        "has_hdd": 0,
        "is_ips": 1,
        "is_touchscreen": 0
    }
    is_valid, errors = validate_laptop_input(sample)
    assert not is_valid
    assert any("RAM" in err for err in errors)
    assert any("company" in err for err in errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation.py`
Expected: FAIL (ModuleNotFoundError: No module named 'src.validation')

- [ ] **Step 3: Implement validation and feature vector construction**

Create `src/validation.py`:
- `validate_laptop_input(input_dict)`: validates presence of required fields, numerical bounds (`ram`: 2..128, `weight`: 0.4..6.0, `inches`: 10.0..19.0, `res_width`: 800..4000, `res_height`: 600..3000, `storage_gb`: 16..4096), categorical choices in allowed lists (`company`, `typename`, `opsys`), boolean values (0 or 1).
- `build_feature_vector(input_dict)`:
  - Computes `ppi = sqrt(res_width**2 + res_height**2) / inches`.
  - Sets one-hot dummy variables: `company_<Company>` (if Company is 'Acer', all `company_*` columns remain 0 as in `drop_first=True`), `typename_<TypeName>` (if '2 in 1 Convertible', all `typename_*` remain 0), and `opsys_<OpSys>` (if 'Android' or 'Unknown', all `opsys_*` remain 0).
  - Creates DataFrame matching exact columns and order of `ALL_FEATURE_COLUMNS`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation.py`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit validation module**

```bash
git add src/validation.py tests/test_validation.py
git commit -m "feat: implement user input validation and feature transformation"
```

---

### Task 4: Model Training, Evaluation, and Serialization Scripts

**Files:**
- Create: `scripts/export_models.py`
- Create: `scripts/generate_evaluation_results.py`
- Create: `scripts/prepare_clustering.py`
- Output: `models/regression/`, `models/classification/`, `models/clustering/`
- Output: `results/regression_metrics.csv`, `results/classification_metrics.csv`, `results/clustering_metrics.json`
- Test: `tests/test_artifacts_generation.py`

**Interfaces:**
- Consumes: `data/regression.csv`, `data/classification.csv`, `data/clustering.csv`, `src.config`
- Produces:
  - Serialized pipeline artifacts:
    - `models/regression/gradient_boosting_tuned.joblib`
    - `models/regression/random_forest_tuned.joblib`
    - `models/regression/linear_regression.joblib`
    - `models/regression/ridge_regression.joblib`
    - `models/regression/knn_tuned.joblib`
    - `models/regression/decision_tree_tuned.joblib`
  - Classification pipeline artifacts:
    - `models/classification/random_forest_classifier.joblib`
    - `models/classification/svc_classifier.joblib`
    - `models/classification/logistic_regression.joblib`
    - `models/classification/knn_classifier.joblib`
    - `models/classification/decision_tree_classifier.joblib`
    - `models/classification/gradient_boosting_classifier.joblib`
  - Clustering artifacts:
    - `models/clustering/scaler.joblib`
    - `models/clustering/kmeans_models.joblib`
    - `models/clustering/pca_model.joblib`
  - Result metrics files:
    - `results/regression_metrics.csv`
    - `results/classification_metrics.csv`
    - `results/clustering_metrics.json`

- [ ] **Step 1: Write the failing test**

Create `tests/test_artifacts_generation.py`:
```python
from pathlib import Path
import pandas as pd
import json
import joblib
from src.config import MODELS_DIR, RESULTS_DIR

def test_results_files_exist():
    reg_csv = RESULTS_DIR / "regression_metrics.csv"
    clf_csv = RESULTS_DIR / "classification_metrics.csv"
    clust_json = RESULTS_DIR / "clustering_metrics.json"

    assert reg_csv.exists(), f"Missing {reg_csv}"
    assert clf_csv.exists(), f"Missing {clf_csv}"
    assert clust_json.exists(), f"Missing {clust_json}"

    reg_df = pd.read_csv(reg_csv)
    assert len(reg_df) == 10
    assert set(["Algorithm", "R2", "RMSE", "MAE"]).issubset(reg_df.columns)

    clf_df = pd.read_csv(clf_csv)
    assert len(clf_df) >= 5
    assert set(["Algorithm", "Accuracy", "Weighted F1"]).issubset(clf_df.columns)

    with open(clust_json) as f:
        clust_data = json.load(f)
    assert "kmeans_elbow" in clust_data
    assert "silhouette_scores" in clust_data
    assert "cluster_profiles" in clust_data

def test_model_pipelines_loadable():
    gb_path = MODELS_DIR / "regression" / "gradient_boosting_tuned.joblib"
    assert gb_path.exists()
    model = joblib.load(gb_path)
    assert hasattr(model, "predict")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_artifacts_generation.py`
Expected: FAIL (files do not exist yet)

- [ ] **Step 3: Implement scripts and generate artifacts**

Implement `scripts/export_models.py`:
- Replicates the 10 regression models and pipelines from `Regression_algorithms_10.ipynb` with exact `random_state=42` and `stratified` split.
- Replicates the classification models from `classification.ipynb` with `random_state=42` and `stratified` split, plus verified additions (Random Forest, Gradient Boosting, AdaBoost, Bagging, MLP).
- Saves pipelines (including `StandardScaler` where appropriate or integrated in `Pipeline`) into `models/regression/` and `models/classification/`.

Implement `scripts/generate_evaluation_results.py`:
- Computes comprehensive evaluation metrics (R2, RMSE, MAE, 5-fold CV for regression; Accuracy, Precision, Recall, Weighted F1, Multiclass ROC-AUC OvR, Confusion Matrix arrays for classification).
- Records data source ("Notebook Historical" vs "Verified Script Extension").
- Writes `results/regression_metrics.csv` and `results/classification_metrics.csv`.

Implement `scripts/prepare_clustering.py`:
- Loads `data/clustering.csv` (strictly 42 features, NO target columns).
- Computes inertia, silhouette score, Davies-Bouldin index, Calinski-Harabasz index for k=2..10.
- Fits KMeans for k=2..6, Agglomerative Hierarchical clustering for k=2..6 across linkages (`ward`, `complete`, `average`).
- Generates 2D PCA and t-SNE coordinates for interactive plotting.
- Computes cluster summary profiles (counts, %, means/medians, and automated descriptive profiles).
- Saves models to `models/clustering/` and results to `results/clustering_metrics.json`.

Execute the scripts to generate the models and metrics.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_artifacts_generation.py`
Expected: PASS

- [ ] **Step 5: Verify notebook hashes are untouched**

Run: `pytest tests/test_audit_and_config.py`
Expected: PASS

- [ ] **Step 6: Commit scripts and reproducible result metrics**

```bash
git add scripts/ results/ tests/test_artifacts_generation.py
git commit -m "feat: implement model export, evaluation, and clustering scripts"
```

---

### Task 5: Model Registry and Inference Layer

**Files:**
- Create: `src/model_registry.py`
- Create: `src/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: `src.config`, `src.validation`, `models/`
- Produces:
  - `src.model_registry.get_available_models(task: str) -> list[str]`
  - `src.model_registry.load_model(task: str, model_name: str) -> object`
  - `src.inference.predict_price(model_name: str, input_dict: dict) -> dict`
  - `src.inference.predict_price_category(model_name: str, input_dict: dict) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference.py`:
```python
from src.model_registry import get_available_models, load_model
from src.inference import predict_price, predict_price_category

def test_available_models():
    reg_models = get_available_models("regression")
    assert len(reg_models) > 0
    assert "Gradient Boosting Regressor (Tuned)" in reg_models or "gradient_boosting_tuned" in reg_models

def test_predict_price():
    sample = {
        "company": "Hp",
        "typename": "Notebook",
        "opsys": "Windows 10",
        "ram": 8.0,
        "weight": 2.0,
        "inches": 15.6,
        "res_width": 1920,
        "res_height": 1080,
        "storage_gb": 256.0,
        "has_ssd": 1,
        "has_hdd": 0,
        "is_ips": 0,
        "is_touchscreen": 0
    }
    models = get_available_models("regression")
    res = predict_price(models[0], sample)
    assert "predicted_price" in res
    assert res["predicted_price"] > 10000
    assert "formatted_price" in res
    assert "₹" in res["formatted_price"]

def test_predict_price_category():
    sample = {
        "company": "Apple",
        "typename": "Ultrabook",
        "opsys": "Macos",
        "ram": 16.0,
        "weight": 1.37,
        "inches": 13.3,
        "res_width": 2560,
        "res_height": 1600,
        "storage_gb": 512.0,
        "has_ssd": 1,
        "has_hdd": 0,
        "is_ips": 1,
        "is_touchscreen": 0
    }
    models = get_available_models("classification")
    res = predict_price_category(models[0], sample)
    assert res["predicted_category"] in ["Budget", "Mid-range", "Premium", "Flagship"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py`
Expected: FAIL (ModuleNotFoundError: No module named 'src.model_registry')

- [ ] **Step 3: Implement model registry and inference layer**

Implement `src/model_registry.py`:
- Cached model loading using joblib.
- Mapping between user-friendly display names and model artifact paths.
- Error handling with clear error messages when an artifact is not found.

Implement `src/inference.py`:
- `predict_price`: Validates input, builds 42-feature vector, calls loaded pipeline `predict`, formats price in INR format (e.g. `₹54,230`), attaches performance metrics and limitations disclaimer.
- `predict_price_category`: Validates input, builds feature vector, calls pipeline `predict`, calls `predict_proba` if supported to return probability dictionary across classes, attaches model accuracy/F1.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference.py`
Expected: PASS

- [ ] **Step 5: Commit inference layer**

```bash
git add src/model_registry.py src/inference.py tests/test_inference.py
git commit -m "feat: implement model registry and inference layer"
```

---

### Task 6: Visualization Module

**Files:**
- Create: `src/visualization.py`
- Test: `tests/test_visualization.py`

**Interfaces:**
- Consumes: Plotly, Pandas, NumPy
- Produces:
  - `plot_model_comparison_bar(df, metric_col, title, higher_is_better=True) -> go.Figure`
  - `plot_actual_vs_predicted(y_true, y_pred, title) -> go.Figure`
  - `plot_residuals(y_true, y_pred, title) -> go.Figure`
  - `plot_confusion_matrix(cm, class_names, title) -> go.Figure`
  - `plot_feature_importance(importance_df, top_n=15) -> go.Figure`
  - `plot_pca_clusters(pca_df, cluster_col, title) -> go.Figure`
  - `plot_clustering_diagnostics(k_values, inertias, silhouettes, db_indices, ch_indices) -> go.Figure`

- [ ] **Step 1: Write the failing test**

Create `tests/test_visualization.py`:
```python
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.visualization import (
    plot_model_comparison_bar,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_confusion_matrix,
    plot_pca_clusters
)

def test_visualization_figures_created():
    df = pd.DataFrame({"Algorithm": ["A", "B"], "R2": [0.8, 0.7]})
    fig = plot_model_comparison_bar(df, "R2", "Test Comparison")
    assert isinstance(fig, go.Figure)

    y_true = np.array([50000, 60000, 70000])
    y_pred = np.array([52000, 58000, 71000])
    fig2 = plot_actual_vs_predicted(y_true, y_pred, "Actual vs Predicted")
    assert isinstance(fig2, go.Figure)

    fig3 = plot_residuals(y_true, y_pred, "Residuals")
    assert isinstance(fig3, go.Figure)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualization.py`
Expected: FAIL (ModuleNotFoundError: No module named 'src.visualization')

- [ ] **Step 3: Implement visualization module**

Create `src/visualization.py` using Plotly with dark theme styling (`plotly_dark`), consistent color palettes, hover templates, and clean responsive layouts.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualization.py`
Expected: PASS

- [ ] **Step 5: Commit visualization module**

```bash
git add src/visualization.py tests/test_visualization.py
git commit -m "feat: implement interactive plotly visualization helpers"
```

---

### Task 7: Landing Page (Laptop Intelligence Dashboard)

**Files:**
- Create: `app.py`
- Test: `tests/test_app_structure.py`

**Interfaces:**
- Consumes: `src.config`, `src.data_loader`, Streamlit
- Produces: Main landing page with project overview, task summaries, quick metrics cards, navigation links, and dataset limitations.

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_structure.py`:
```python
from pathlib import Path
from src.config import BASE_DIR

def test_pages_exist():
    app_file = BASE_DIR / "app.py"
    assert app_file.exists()

    pages_dir = BASE_DIR / "pages"
    assert pages_dir.exists()

    expected_pages = [
        "1_Price_Predictor.py",
        "2_Price_Classification.py",
        "3_Laptop_Clustering.py",
        "4_Model_Laboratory.py"
    ]
    for p in expected_pages:
        assert (pages_dir / p).exists(), f"Page {p} missing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_structure.py`
Expected: FAIL (app.py and pages do not exist)

- [ ] **Step 3: Implement `app.py`**

Build `app.py`:
- Page title: "Laptop Intelligence Dashboard"
- High-level capstone description: ML tasks (Regression, Classification, Clustering).
- Quick KPI metrics cards showing top verified models and best metrics (Gradient Boosting R² = 0.8121, Top Classifier Accuracy/F1, Clustering segments).
- Navigation cards pointing to each of the 4 sub-pages with descriptions.
- Technical overview of the dataset (1,243 laptops, 42 features) and essential limitations disclaimer (dataset collected from specific retail snapshots, currency in INR, price capped by IQR).

- [ ] **Step 4: Create dummy page placeholders to verify structure passes**

Create `pages/1_Price_Predictor.py`, `pages/2_Price_Classification.py`, `pages/3_Laptop_Clustering.py`, `pages/4_Model_Laboratory.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_app_structure.py`
Expected: PASS

- [ ] **Step 6: Commit main application landing page**

```bash
git add app.py pages/ tests/test_app_structure.py
git commit -m "feat: implement landing page and multi-page structure"
```

---

### Task 8: Page 1 - Laptop Price Predictor (Regression)

**Files:**
- Modify: `pages/1_Price_Predictor.py`
- Consumes: `src.validation`, `src.inference`, `src.model_registry`
- Produces: Complete interactive regression interface.

- [ ] **Step 1: Write Streamlit app unit test**

Create `tests/test_page_1.py`:
```python
from streamlit.testing.v1 import AppTest

def test_page_1_renders():
    at = AppTest.from_file("pages/1_Price_Predictor.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) > 0 or len(at.header) > 0
```

- [ ] **Step 2: Run test to verify initial state or failure**

Run: `pytest tests/test_page_1.py`

- [ ] **Step 3: Implement Page 1**

In `pages/1_Price_Predictor.py`:
- Form inputs:
  - Brand (Company dropdown: Dell, HP, Lenovo, Asus, Apple, Acer, etc.)
  - Type (TypeName dropdown: Notebook, Gaming, Ultrabook, 2 in 1 Convertible, Workstation, Netbook)
  - Operating System (Windows 10, macOS, Linux, Chrome OS, etc.)
  - RAM (slider/select: 2 to 64 GB)
  - Weight in kg (number input: 0.8 to 4.5 kg)
  - Screen size (inches: 11.6, 13.3, 14.0, 15.6, 17.3, etc.)
  - Screen resolution (dropdown: 1920x1080, 1366x768, 2560x1440, 3840x2160, 2560x1600, etc.)
  - Display features (Checkboxes: IPS Panel, Touchscreen)
  - Storage (Storage GB: 128, 256, 512, 1024, 2048; Checkboxes: SSD, HDD)
- Model Selector: Choose between available regression models (Gradient Boosting Tuned, Random Forest Tuned, Ridge, etc.).
- "Predict Price" button:
  - Validates input.
  - Builds 42-column vector with pre-fitted transformations.
  - Displays predicted price in Indian Rupees (e.g. `₹62,450`).
  - Displays model metrics badge (R²: 0.8121, RMSE: ₹14,868, MAE: ₹11,025).
  - Provides a "Compare Configurations" mode allowing side-by-side spec comparison and price delta calculation.
  - Displays prediction limitations note.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_page_1.py`
Expected: PASS

- [ ] **Step 5: Commit Page 1**

```bash
git add pages/1_Price_Predictor.py tests/test_page_1.py
git commit -m "feat: implement Page 1 Laptop Price Predictor"
```

---

### Task 9: Page 2 - Laptop Price Classification

**Files:**
- Modify: `pages/2_Price_Classification.py`
- Consumes: `src.validation`, `src.inference`, `src.model_registry`, `src.visualization`, `src.data_loader`
- Produces: Complete interactive classification interface.

- [ ] **Step 1: Write Streamlit app unit test**

Create `tests/test_page_2.py`:
```python
from streamlit.testing.v1 import AppTest

def test_page_2_renders():
    at = AppTest.from_file("pages/2_Price_Classification.py", default_timeout=30)
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_page_2.py`

- [ ] **Step 3: Implement Page 2**

In `pages/2_Price_Classification.py`:
- Interactive laptop spec inputs for classification.
- Model selector: Select from verified trained classification models (Random Forest, SVC, Logistic Regression, KNN, Decision Tree, Gradient Boosting).
- Prediction section:
  - Predicts price tier (`Budget`, `Mid-range`, `Premium`, `Flagship`).
  - Visualizes class probabilities as a Plotly horizontal bar chart (for models supporting `predict_proba`).
  - Clear explanation of what each price tier represents based on dataset quartiles (`Budget`: ≤ ₹32,650, `Mid-range`: ₹32,650–₹52,693, `Premium`: ₹52,693–₹79,733, `Flagship`: > ₹79,733).
- Model comparison leaderboard:
  - Shows verified models, Accuracy, Precision, Recall, Weighted F1, Multiclass ROC-AUC (OvR).
  - Displays interactive confusion matrix for selected model.
  - Displays class balance chart.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_page_2.py`
Expected: PASS

- [ ] **Step 5: Commit Page 2**

```bash
git add pages/2_Price_Classification.py tests/test_page_2.py
git commit -m "feat: implement Page 2 Laptop Price Classification"
```

---

### Task 10: Page 3 - Laptop Clustering and Segmentation

**Files:**
- Modify: `pages/3_Laptop_Clustering.py`
- Consumes: `src.data_loader`, `src.visualization`, `results/clustering_metrics.json`
- Produces: Interactive unsupervised learning and market segmentation page.

- [ ] **Step 1: Write Streamlit app unit test**

Create `tests/test_page_3.py`:
```python
from streamlit.testing.v1 import AppTest

def test_page_3_renders():
    at = AppTest.from_file("pages/3_Laptop_Clustering.py", default_timeout=30)
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_page_3.py`

- [ ] **Step 3: Implement Page 3**

In `pages/3_Laptop_Clustering.py`:
- Prominent statement and code guarantee: **Zero Target Leakage** — Clustering is performed exclusively on the 42 hardware feature columns without `price` or `price_category`.
- Algorithm & Cluster Selection:
  - Choose between K-Means and Agglomerative Hierarchical Clustering.
  - Select number of clusters $k$ (2 to 6).
- Evaluation & Diagnostics:
  - Elbow Curve (Inertia vs $k$)
  - Silhouette Score curve
  - Davies-Bouldin Index (lower is better) & Calinski-Harabasz Index (higher is better)
  - Hierarchical linkage comparison (Ward vs Complete vs Average)
- 2D Cluster Visualizations:
  - Interactive 2D PCA projection scatter plot with cluster color-coding and hover inspection (RAM, Weight, Inches, Company, TypeName).
  - 2D t-SNE projection scatter plot.
- Cluster Profiles:
  - Cluster size (count and percentage).
  - Numerical feature averages (RAM, weight, storage, inches, PPI).
  - Dominant laptop categories and brands per cluster.
  - Automated descriptive summary per segment (e.g. "Cluster 1: Ultraportable Premium Laptops (Lightweight, High PPI, SSD)", "Cluster 2: Heavy Gaming & Workstations (High RAM, Dedicated Storage)").

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_page_3.py`
Expected: PASS

- [ ] **Step 5: Commit Page 3**

```bash
git add pages/3_Laptop_Clustering.py tests/test_page_3.py
git commit -m "feat: implement Page 3 Laptop Clustering and Segmentation"
```

---

### Task 11: Page 4 - Model Laboratory

**Files:**
- Modify: `pages/4_Model_Laboratory.py`
- Consumes: `src.data_loader`, `src.visualization`, `results/`
- Produces: Dedicated model comparison, leaderboard, and evaluation diagnostics center.

- [ ] **Step 1: Write Streamlit app unit test**

Create `tests/test_page_4.py`:
```python
from streamlit.testing.v1 import AppTest

def test_page_4_renders():
    at = AppTest.from_file("pages/4_Model_Laboratory.py", default_timeout=30)
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_page_4.py`

- [ ] **Step 3: Implement Page 4**

In `pages/4_Model_Laboratory.py`:
- Tabbed interface:
  - **Tab 1: Regression Analysis**:
    - Official 10-model leaderboard ranking by R², RMSE, MAE.
    - Top models 5-Fold Cross Validation comparison.
    - Actual vs. Predicted scatter plot with perfect prediction line.
    - Residuals distribution and Residuals vs. Fitted plot.
    - Feature Importance plot (Random Forest / Gradient Boosting) highlighting RAM, Weight, TypeName, and PPI as dominant drivers.
  - **Tab 2: Classification Analysis**:
    - Complete classification leaderboard (Accuracy, Precision, Recall, Weighted F1, Multiclass ROC-AUC OvR).
    - Multi-class ROC Curves.
    - Confusion matrices for all evaluated models.
    - Class balance analysis.
  - **Tab 3: Clustering Analysis**:
    - Diagnostic score comparison table across $k=2..6$.
    - Linkage evaluation for hierarchical clustering.
    - Cluster silhouette distribution.
  - **Tab 4: Provenance & Data Integrity**:
    - Transparent documentation of metric sources: distinguishes original notebook results from script-extended results.
    - Dataset splits specification (`test_size=0.20, random_state=42`).
    - Verification of notebook hashes and read-only status.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_page_4.py`
Expected: PASS

- [ ] **Step 5: Commit Page 4**

```bash
git add pages/4_Model_Laboratory.py tests/test_page_4.py
git commit -m "feat: implement Page 4 Model Laboratory"
```

---

### Task 12: Documentation, Deployment Readiness, and Final Verification

**Files:**
- Modify: `README.md` (preserve existing content, append new application documentation)
- Test: All unit and page tests (`pytest`)
- Audit: Verify notebook hashes

- [ ] **Step 1: Update README.md with comprehensive application documentation**

Add to `README.md`:
- Overview of the Laptop Intelligence Dashboard web application.
- Architecture and module structure.
- Local setup instructions and run commands (`streamlit run app.py`).
- How to reproduce model artifacts using `scripts/export_models.py`, `scripts/generate_evaluation_results.py`, and `scripts/prepare_clustering.py`.
- Model leaderboard and verified performance summary.
- Testing instructions (`pytest`).
- Free-tier deployment instructions (Streamlit Community Cloud and Hugging Face Spaces).
- Notebook integrity statement and hash verification.

- [ ] **Step 2: Run complete test suite**

Run: `pytest -v`
Expected: All tests pass.

- [ ] **Step 3: Verify Notebook SHA-256 Hashes**

Run verification check to ensure no notebook has been modified, renamed, moved, or altered in any byte.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: document application setup, reproduction, and deployment"
```

---
