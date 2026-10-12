from pathlib import Path
import streamlit as st
import joblib
import pandas as pd

from src.config import MODELS_DIR, RESULTS_DIR

REGRESSION_MODEL_FILENAMES = {
    "Gradient Boosting Regressor (Tuned)": "gradient_boosting_tuned.joblib",
    "Random Forest Regressor (Tuned)": "random_forest_tuned.joblib",
    "Polynomial Regression (degree 2)": "polynomial_regression_degree_2.joblib",
    "K-Nearest Neighbors Regressor (Tuned)": "knn_tuned.joblib",
    "Decision Tree Regressor (Tuned)": "decision_tree_tuned.joblib",
    "ElasticNet Regression": "elasticnet_regression.joblib",
    "Ridge Regression": "ridge_regression.joblib",
    "Lasso Regression": "lasso_regression.joblib",
    "Linear Regression": "linear_regression.joblib",
    "Support Vector Regressor (Tuned)": "svr_tuned.joblib",
}

CLASSIFICATION_MODEL_FILENAMES = {
    "Random Forest": "random_forest_classifier.joblib",
    "Gradient Boosting": "gradient_boosting_classifier.joblib",
    "Bagging": "bagging_classifier.joblib",
    "SVC": "svc_classifier.joblib",
    "MLP Classifier": "mlp_classifier.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "AdaBoost": "adaboost_classifier.joblib",
    "KNN": "knn_classifier.joblib",
    "Decision Tree": "decision_tree_classifier.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
}

def get_available_models(task: str) -> list[str]:
    """Returns the list of available model names for the specified task."""
    if task == "regression":
        available = []
        for name, filename in REGRESSION_MODEL_FILENAMES.items():
            if (MODELS_DIR / "regression" / filename).exists():
                available.append(name)
        return available
    elif task == "classification":
        available = []
        for name, filename in CLASSIFICATION_MODEL_FILENAMES.items():
            if (MODELS_DIR / "classification" / filename).exists():
                available.append(name)
        return available
    elif task == "clustering":
        models_file = MODELS_DIR / "clustering" / "kmeans_models.joblib"
        return ["K-Means", "Agglomerative Hierarchical"] if models_file.exists() else []
    else:
        raise ValueError(f"Unknown task: '{task}'")


@st.cache_resource
def load_model(task: str, model_name: str):
    """
    Loads and caches a trained model pipeline from disk using Joblib.
    Raises FileNotFoundError if the artifact is missing.
    """
    if task == "regression":
        filename = REGRESSION_MODEL_FILENAMES.get(model_name)
        if not filename:
            raise ValueError(f"Model '{model_name}' not recognized for regression.")
        path = MODELS_DIR / "regression" / filename
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}. Run scripts/export_models.py.")
        return joblib.load(path)

    elif task == "classification":
        if model_name == "Multi-layer Perceptron":
            model_name = "MLP Classifier"
        filename = CLASSIFICATION_MODEL_FILENAMES.get(model_name)
        if not filename:
            raise ValueError(f"Model '{model_name}' not recognized for classification.")
        path = MODELS_DIR / "classification" / filename
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}. Run scripts/export_models.py.")
        return joblib.load(path)

    elif task == "clustering":
        if model_name == "scaler":
            path = MODELS_DIR / "clustering" / "scaler.joblib"
        elif model_name == "pca":
            path = MODELS_DIR / "clustering" / "pca_model.joblib"
        elif model_name == "kmeans_dict":
            path = MODELS_DIR / "clustering" / "kmeans_models.joblib"
        else:
            raise ValueError(f"Unknown clustering artifact '{model_name}'.")
        if not path.exists():
            raise FileNotFoundError(f"Clustering artifact not found at {path}. Run scripts/prepare_clustering.py.")
        return joblib.load(path)

    else:
        raise ValueError(f"Unknown task: '{task}'")
