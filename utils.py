"""Shared helpers: preprocessing, model definitions and metric calculation."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "Revenue"
CATEGORICAL = ["Month", "VisitorType"]          # text columns -> one-hot
NUMERIC = ["Administrative", "Administrative_Duration", "Informational",
           "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
           "BounceRates", "ExitRates", "PageValues", "SpecialDay",
           "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}


def clean_dataframe(df):
    """Make Weekend / Revenue numeric (0/1) whether they are True/False, 'TRUE' or 0/1."""
    df = df.copy()
    for col in ("Weekend", TARGET):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}).astype(int)
    return df


def _preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def build_models():
    """5 full pipelines: preprocessing + classifier (so raw CSV columns can be fed in)."""
    def make(clf):
        return Pipeline([("prep", _preprocessor()), ("clf", clf)])
    return {
        "Logistic Regression": make(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        "Decision Tree": make(DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)),
        "kNN": make(KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": make(GaussianNB()),
        "Support Vector Machine": make(SVC(kernel="rbf", class_weight="balanced",
                                           probability=True, random_state=42)),
    }


def compute_metrics(model, X, y):
    """Return (metrics dict, predictions). Positive class = 1 (purchase)."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred
