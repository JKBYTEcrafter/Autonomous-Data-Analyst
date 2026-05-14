"""
AutoML Engine — pure scikit-learn implementation.
Replaces PyCaret to avoid the heavy dependency and installation issues.
Supports: Binary Classification, Multi-Class Classification, Regression, Clustering.
"""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    silhouette_score,
)
from sklearn.cluster import KMeans

# --- Classifiers ---
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# --- Regressors ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(_BASE_DIR, "app", "models")
sys.path.insert(0, _BASE_DIR)

from app.utils.eda import detect_problem_type
from app.utils.data_ingestion import load_dataset
from app.utils.data_cleaning import handle_missing_values, encode_categorical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess(df: pd.DataFrame, target_col: str):
    """Basic preprocessing: fill NaNs, encode categoricals, return X, y arrays."""
    df = df.copy()
    y_raw = df.pop(target_col)

    df = handle_missing_values(df, strategy="mean")
    df = encode_categorical(df, strategy="label")
    X = df.select_dtypes(include=["number"]).values.astype(float)

    return X, y_raw


def _sanitize_leaderboard(records: list) -> list:
    safe = []
    for row in records:
        safe_row = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                safe_row[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                safe_row[k] = None if (np.isnan(v) or np.isinf(v)) else round(float(v), 4)
            else:
                safe_row[k] = v
        safe.append(safe_row)
    return safe


def _generate_confusion_matrix_chart(y_true, y_pred) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, text_auto=True, aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    return fig.to_json()


def _generate_roc_curve_chart(y_true, y_score, problem_type: str) -> str:
    try:
        if "Binary" in problem_type:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"ROC (AUC={auc:.3f})",
                                     line=dict(color="#4ECDC4", width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Random", line=dict(color="gray", dash="dash")))
            fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                               template="plotly_dark", paper_bgcolor="#0e1117")
            return fig.to_json()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _clf_candidates():
    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Ridge Classifier", RidgeClassifier()),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=50, random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Naive Bayes", GaussianNB()),
        ("SVM", SVC(probability=True, random_state=42)),
    ]
    if _HAS_XGB:
        candidates.append(("XGBoost", XGBClassifier(n_estimators=100, random_state=42,
                                                     eval_metric="logloss", verbosity=0)))
    if _HAS_LGB:
        candidates.append(("LightGBM", LGBMClassifier(n_estimators=100, random_state=42,
                                                        verbose=-1)))
    return candidates


def _run_classification(df: pd.DataFrame, target_col: str, problem_type: str) -> dict:
    X, y_raw = _preprocess(df, target_col)

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metric = "accuracy"

    leaderboard = []
    best_score = -np.inf
    best_model = None
    best_name = ""

    for name, model in _clf_candidates():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            mean_acc = float(np.mean(scores))
            std_acc = float(np.std(scores))
            leaderboard.append({"Model": name, "Accuracy": round(mean_acc, 4),
                                 "Std": round(std_acc, 4)})
            if mean_acc > best_score:
                best_score = mean_acc
                best_model = model
                best_name = name
        except Exception:
            pass

    leaderboard.sort(key=lambda r: r["Accuracy"], reverse=True)

    # Fit the best model on full data
    best_model.fit(X, y)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_filename = f"{problem_type.replace(' ', '_').lower()}_best_model.pkl"
    save_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump({"model": best_model, "label_encoder": le}, save_path)

    # Confusion matrix
    y_pred = best_model.predict(X)
    confusion_chart = _generate_confusion_matrix_chart(y, y_pred)

    # ROC curve (binary only)
    roc_chart = ""
    if "Binary" in problem_type and hasattr(best_model, "predict_proba"):
        try:
            y_score = best_model.predict_proba(X)[:, 1]
            roc_chart = _generate_roc_curve_chart(y, y_score, problem_type)
        except Exception:
            pass

    return {
        "problem_type": problem_type,
        "leaderboard": _sanitize_leaderboard(leaderboard),
        "best_model_path": save_path,
        "confusion_matrix": confusion_chart,
        "roc_curve": roc_chart,
    }


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _reg_candidates():
    candidates = [
        ("Linear Regression", LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso(max_iter=5000)),
        ("ElasticNet", ElasticNet(max_iter=5000)),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("AdaBoost", AdaBoostRegressor(n_estimators=50, random_state=42)),
        ("K-Nearest Neighbors", KNeighborsRegressor()),
        ("SVR", SVR()),
    ]
    if _HAS_XGB:
        candidates.append(("XGBoost", XGBRegressor(n_estimators=100, random_state=42, verbosity=0)))
    if _HAS_LGB:
        candidates.append(("LightGBM", LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)))
    return candidates


def _run_regression(df: pd.DataFrame, target_col: str) -> dict:
    X, y_raw = _preprocess(df, target_col)
    y = pd.to_numeric(y_raw, errors="coerce").fillna(0).values.astype(float)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    leaderboard = []
    best_score = np.inf
    best_model = None

    for name, model in _reg_candidates():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
            rmse = float(np.sqrt(-np.mean(scores)))
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
            r2 = float(np.mean(r2_scores))
            leaderboard.append({"Model": name, "RMSE": round(rmse, 4), "R2": round(r2, 4)})
            if rmse < best_score:
                best_score = rmse
                best_model = model
        except Exception:
            pass

    leaderboard.sort(key=lambda r: r["RMSE"])

    best_model.fit(X, y)
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "regression_best_model.pkl")
    joblib.dump(best_model, save_path)

    return {
        "problem_type": "Regression",
        "leaderboard": _sanitize_leaderboard(leaderboard),
        "best_model_path": save_path,
        "confusion_matrix": "",
        "roc_curve": "",
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _run_clustering(df: pd.DataFrame) -> dict:
    df_clean = handle_missing_values(df.copy(), strategy="mean")
    df_clean = encode_categorical(df_clean, strategy="label")
    num_df = df_clean.select_dtypes(include=["number"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_df)

    scores = []
    k_range = range(2, min(9, len(df_clean) // 10 + 2))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores.append({
            "Model": f"KMeans (k={k})",
            "Silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
            "Inertia": round(float(km.inertia_), 2),
        })

    best_entry = max(scores, key=lambda x: x["Silhouette"])
    best_k_val = int(best_entry["Model"].split("=")[1].rstrip(")"))
    best_model_obj = KMeans(n_clusters=best_k_val, random_state=42, n_init=10)
    best_model_obj.fit(X_scaled)

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "clustering_kmeans_model.pkl")
    joblib.dump(best_model_obj, save_path)

    return {
        "problem_type": "Clustering",
        "leaderboard": _sanitize_leaderboard(scores),
        "best_model_path": save_path,
        "confusion_matrix": "",
        "roc_curve": "",
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_automl_pipeline(file_path: str, target_col: str, problem_type: str = None) -> dict:
    df = load_dataset(file_path)

    if problem_type is None:
        problem_type = detect_problem_type(df, target_col)

    if "Classification" in problem_type:
        return _run_classification(df, target_col, problem_type)
    elif "Regression" in problem_type:
        return _run_regression(df, target_col)
    elif problem_type == "Clustering":
        return _run_clustering(df)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
