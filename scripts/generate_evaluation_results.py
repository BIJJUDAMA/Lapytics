"""
Evaluation Results Generator
Generates verifiable metrics files (results/regression_metrics.csv, results/classification_metrics.csv)
recording actual test performance, cross-validation, and provenance.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import label_binarize
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE, PRICE_CATEGORIES

def evaluate_regression():
    print("Generating regression evaluation metrics...")
    df = pd.read_csv(DATA_DIR / "regression.csv")
    X = df.drop(columns=["price"]).copy()
    y = df["price"].copy()

    target_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=target_bins
    )

    models_map = {
        "Gradient Boosting Regressor (Tuned)": ("gradient_boosting_tuned.joblib", "Notebook Verified"),
        "Random Forest Regressor (Tuned)": ("random_forest_tuned.joblib", "Notebook Verified"),
        "Polynomial Regression (degree 2)": ("polynomial_regression_degree_2.joblib", "Notebook Verified"),
        "K-Nearest Neighbors Regressor (Tuned)": ("knn_tuned.joblib", "Notebook Verified"),
        "Decision Tree Regressor (Tuned)": ("decision_tree_tuned.joblib", "Notebook Verified"),
        "ElasticNet Regression": ("elasticnet_regression.joblib", "Notebook Verified"),
        "Ridge Regression": ("ridge_regression.joblib", "Notebook Verified"),
        "Lasso Regression": ("lasso_regression.joblib", "Notebook Verified"),
        "Linear Regression": ("linear_regression.joblib", "Notebook Verified"),
        "Support Vector Regressor (Tuned)": ("svr_tuned.joblib", "Notebook Verified"),
    }

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, (filename, _) in models_map.items():
        model_path = MODELS_DIR / "regression" / filename
        if not model_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # 5-fold CV on train
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)

        results.append({
            "Algorithm": name,
            "R2": round(r2, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "CV_R2_Mean": round(cv_scores.mean(), 4),
            "CV_R2_Std": round(cv_scores.std(), 4),
        })

    res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    res_df.insert(0, "Rank", np.arange(1, len(res_df) + 1))
    out_path = RESULTS_DIR / "regression_metrics.csv"
    res_df.to_csv(out_path, index=False)
    print(f"Saved regression metrics to {out_path}")
    print(res_df.to_string(index=False))


def evaluate_classification():
    print("\nGenerating classification evaluation metrics...")
    df = pd.read_csv(DATA_DIR / "classification.csv")
    X = df.drop(columns=["price_category"]).copy()
    y = df["price_category"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # All 10 models verified from classification_10_algorithms.ipynb (Cell 58)
    models_map = {
        "Random Forest": ("random_forest_classifier.joblib", "Notebook Verified"),
        "Gradient Boosting": ("gradient_boosting_classifier.joblib", "Notebook Verified"),
        "Bagging": ("bagging_classifier.joblib", "Notebook Verified"),
        "SVC": ("svc_classifier.joblib", "Notebook Verified"),
        "MLP Classifier": ("mlp_classifier.joblib", "Notebook Verified"),
        "Logistic Regression": ("logistic_regression.joblib", "Notebook Verified"),
        "AdaBoost": ("adaboost_classifier.joblib", "Notebook Verified"),
        "KNN": ("knn_classifier.joblib", "Notebook Verified"),
        "Decision Tree": ("decision_tree_classifier.joblib", "Notebook Verified"),
        "Gaussian Naive Bayes": ("gaussian_nb.joblib", "Notebook Verified"),
    }

    classes = PRICE_CATEGORIES
    y_test_bin = label_binarize(y_test, classes=classes)

    results = []
    cms = {}

    for name, (filename, _) in models_map.items():
        model_path = MODELS_DIR / "classification" / filename
        if not model_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC-AUC OvR if probability available
        roc_auc_val = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                roc_auc_val = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted")
            except Exception:
                pass

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cms[name] = cm.tolist()

        results.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Weighted_F1": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "ROC_AUC": round(roc_auc_val, 4) if roc_auc_val is not None else "N/A",
        })

    res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    res_df.insert(0, "Rank", np.arange(1, len(res_df) + 1))
    out_path = RESULTS_DIR / "classification_metrics.csv"
    res_df.to_csv(out_path, index=False)
    print(f"Saved classification metrics to {out_path}")
    print(res_df.to_string(index=False))

    # Save confusion matrices to json
    cm_path = RESULTS_DIR / "classification_confusion_matrices.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "confusion_matrices": cms}, f, indent=2)
    print(f"Saved confusion matrices to {cm_path}")

if __name__ == "__main__":
    evaluate_regression()
    evaluate_classification()
