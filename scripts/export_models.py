"""
Model Training and Export Script
Reproduces and exports verified regression and classification models into models/
without modifying any existing notebook.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, MODELS_DIR, RANDOM_STATE

def export_regression_models():
    print("Training and exporting regression models...")
    df = pd.read_csv(DATA_DIR / "regression.csv")
    X = df.drop(columns=["price"]).copy()
    y = df["price"].copy()

    # Stratified split matching Regression_algorithms_10.ipynb
    target_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=target_bins
    )

    reg_dir = MODELS_DIR / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gradient Boosting Regressor (Tuned) - Rank 1
    gb = GradientBoostingRegressor(learning_rate=0.1, n_estimators=200, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    joblib.dump(gb, reg_dir / "gradient_boosting_tuned.joblib")

    # 2. Random Forest Regressor (Tuned) - Rank 2
    rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, reg_dir / "random_forest_tuned.joblib")

    # 3. Linear Regression
    lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lr.fit(X_train, y_train)
    joblib.dump(lr, reg_dir / "linear_regression.joblib")

    # 4. Ridge Regression
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    joblib.dump(ridge, reg_dir / "ridge_regression.joblib")

    # 5. Lasso Regression
    lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=100.0, max_iter=20000))])
    lasso.fit(X_train, y_train)
    joblib.dump(lasso, reg_dir / "lasso_regression.joblib")

    # 6. ElasticNet Regression
    enet = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000))])
    enet.fit(X_train, y_train)
    joblib.dump(enet, reg_dir / "elasticnet_regression.joblib")

    # 7. Decision Tree Regressor (Tuned: max_depth=7)
    dt = DecisionTreeRegressor(max_depth=7, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    joblib.dump(dt, reg_dir / "decision_tree_tuned.joblib")

    # 8. SVR (Tuned: C=100.0, kernel='linear')
    svr = Pipeline([("scaler", StandardScaler()), ("model", SVR(C=100.0, kernel="linear"))])
    svr.fit(X_train, y_train)
    joblib.dump(svr, reg_dir / "svr_tuned.joblib")

    # 9. KNN Regressor (Tuned: k=3)
    knn = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=3))])
    knn.fit(X_train, y_train)
    joblib.dump(knn, reg_dir / "knn_tuned.joblib")

    # 10. Polynomial Regression (degree 2 on RAM)
    other_cols = [c for c in X.columns if c != "ram"]
    poly_preprocessor = ColumnTransformer([
        ("ram_poly", Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("scaler", StandardScaler())]), ["ram"]),
        ("others", StandardScaler(), other_cols)
    ])
    poly_pipe = Pipeline([("preprocessing", poly_preprocessor), ("model", LinearRegression())])
    poly_pipe.fit(X_train, y_train)
    joblib.dump(poly_pipe, reg_dir / "polynomial_regression_degree_2.joblib")

    print(f"Exported 10 regression models to {reg_dir}")


def export_classification_models():
    print("Training and exporting classification models matching classification_10_algorithms.ipynb...")
    df = pd.read_csv(DATA_DIR / "classification.csv")
    X = df.drop(columns=["price_category"]).copy()
    y = df["price_category"].copy()

    # Stratified split matching classification_10_algorithms.ipynb (test_size=0.20, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    clf_dir = MODELS_DIR / "classification"
    clf_dir.mkdir(parents=True, exist_ok=True)

    # 1. Random Forest Classifier (Best in notebook: n_estimators=100, max_depth=None, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, clf_dir / "random_forest_classifier.joblib")

    # 2. Gradient Boosting Classifier (Best in notebook: n_estimators=200, learning_rate=0.2, max_depth=3, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, max_depth=3, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    joblib.dump(gb, clf_dir / "gradient_boosting_classifier.joblib")

    # 3. Bagging Classifier (Best in notebook: n_estimators=100, base_estimator=DecisionTreeClassifier(random_state=42), random_state=42)
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=100,
        random_state=RANDOM_STATE
    )
    bag.fit(X_train, y_train)
    joblib.dump(bag, clf_dir / "bagging_classifier.joblib")

    # 4. Support Vector Classifier (Best in notebook: C=10, kernel='rbf', probability=True, scaled)
    svc = Pipeline([("scaler", StandardScaler()), ("model", SVC(C=10, kernel="rbf", probability=True, random_state=RANDOM_STATE))])
    svc.fit(X_train, y_train)
    joblib.dump(svc, clf_dir / "svc_classifier.joblib")

    # 5. Multi-layer Perceptron (MLP) (Best in notebook: hidden_layer_sizes=(50,), alpha=0.0001, activation='relu', max_iter=500, scaled)
    mlp = Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(50,), alpha=0.0001, activation='relu', max_iter=500, random_state=RANDOM_STATE))])
    mlp.fit(X_train, y_train)
    joblib.dump(mlp, clf_dir / "mlp_classifier.joblib")

    # 6. Logistic Regression (max_iter=1000, random_state=42, scaled)
    logistic = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    logistic.fit(X_train, y_train)
    joblib.dump(logistic, clf_dir / "logistic_regression.joblib")

    # 7. AdaBoost Classifier (Best in notebook: n_estimators=200, learning_rate=1.0, random_state=42)
    ada = AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=RANDOM_STATE)
    ada.fit(X_train, y_train)
    joblib.dump(ada, clf_dir / "adaboost_classifier.joblib")

    # 8. KNN Classifier (Best in notebook: n_neighbors=3, metric='euclidean', scaled)
    knn = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=3, metric="euclidean"))])
    knn.fit(X_train, y_train)
    joblib.dump(knn, clf_dir / "knn_classifier.joblib")

    # 9. Decision Tree Classifier (Best in notebook: max_depth=6, random_state=42)
    dt = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    joblib.dump(dt, clf_dir / "decision_tree_classifier.joblib")

    # 10. Gaussian Naive Bayes (scaled)
    gnb = Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())])
    gnb.fit(X_train, y_train)
    joblib.dump(gnb, clf_dir / "gaussian_nb.joblib")

    print(f"Exported all 10 notebook-aligned classification models to {clf_dir}")

if __name__ == "__main__":
    export_regression_models()
    export_classification_models()
