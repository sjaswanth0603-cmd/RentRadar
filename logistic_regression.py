import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from preprocessing import preprocess_classification_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_METRICS_PATH = os.path.join(MODELS_DIR, "logreg_metrics.pkl")

TITLE_COLOR = "#1f3a5f"
LABEL_COLOR = "#4a5568"


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def compute_classification_metrics(y_true, y_pred, y_train=None, y_train_pred=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred) if y_train is not None else acc
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "train_accuracy": round(float(train_acc) * 100, 2),
        "test_accuracy": round(float(acc) * 100, 2),
        "precision_macro": round(float(prec_macro) * 100, 2),
        "precision_weighted": round(float(prec_weighted) * 100, 2),
        "recall_macro": round(float(rec_macro) * 100, 2),
        "recall_weighted": round(float(rec_weighted) * 100, 2),
        "f1_macro": round(float(f1_macro) * 100, 2),
        "f1_weighted": round(float(f1_weighted) * 100, 2),
    }


def run_logistic_regression(force_retrain: bool = False) -> dict:
    """
    Multi-class Logistic Regression across 3 Quantile Price Tiers:
    Budget (< $1,130), Mid-Range ($1,130-$1,605), Premium (> $1,605).
    Compares:
    1. Unregularized Logistic Regression (None)
    2. L2 Regularization (Ridge Logistic)
    3. L1 Regularization (Lasso Logistic)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    if not force_retrain and os.path.exists(CACHE_METRICS_PATH):
        try:
            with open(CACHE_METRICS_PATH, "rb") as f:
                cached_data = pickle.load(f)
            if os.path.exists(_chart_path("logistic_regression_confusion_matrix.png")):
                return cached_data
        except Exception:
            pass

    print("[LogisticRegression] Preprocessing classification dataset (3 Quantile Tiers)...")
    X_train, X_test, y_train, y_test, preprocessor, thresholds = preprocess_classification_data()
    class_labels = ["Budget", "Mid-Range", "Premium"]

    # Fast representative sample for L1 SAGA solver
    n_train = X_train.shape[0]
    np.random.seed(42)
    sub_l1_idx = np.random.choice(n_train, size=min(10000, n_train), replace=False)
    X_train_l1 = X_train[sub_l1_idx]
    y_train_l1 = y_train.iloc[sub_l1_idx] if hasattr(y_train, "iloc") else y_train[sub_l1_idx]

    print("[LogisticRegression] Training Model 1: Without Regularization...")
    # 1. Unregularized
    model_unreg = LogisticRegression(penalty=None, solver="lbfgs", max_iter=200, random_state=42)
    model_unreg.fit(X_train_l1, y_train_l1)
    y_pred_unreg = model_unreg.predict(X_test)
    y_pred_train_unreg = model_unreg.predict(X_train_l1)
    metrics_unreg = compute_classification_metrics(y_test, y_pred_unreg, y_train_l1, y_pred_train_unreg)

    print("[LogisticRegression] Training Model 2: L2 Regularization (Ridge)...")
    # 2. L2 Regularization
    model_l2 = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=250, random_state=42)
    model_l2.fit(X_train, y_train)
    y_pred_l2 = model_l2.predict(X_test)
    y_pred_train_l2 = model_l2.predict(X_train)
    metrics_l2 = compute_classification_metrics(y_test, y_pred_l2, y_train, y_pred_train_l2)

    print("[LogisticRegression] Training Model 3: L1 Regularization (Lasso)...")
    # 3. L1 Regularization
    model_l1 = LogisticRegression(penalty="l1", solver="saga", l1_ratio=1.0, C=0.8, max_iter=60, tol=1e-2, random_state=42)
    model_l1.fit(X_train_l1, y_train_l1)
    y_pred_l1 = model_l1.predict(X_test)
    y_pred_train_l1 = model_l1.predict(X_train_l1)
    metrics_l1 = compute_classification_metrics(y_test, y_pred_l1, y_train_l1, y_pred_train_l1)

    # Detailed classification report for best model (L2)
    report_dict = classification_report(y_test, y_pred_l2, target_names=class_labels, output_dict=True, zero_division=0)
    per_class_metrics = []
    for lbl in class_labels:
        if lbl in report_dict:
            per_class_metrics.append({
                "class": lbl,
                "precision": round(report_dict[lbl]["precision"] * 100, 1),
                "recall": round(report_dict[lbl]["recall"] * 100, 1),
                "f1": round(report_dict[lbl]["f1-score"] * 100, 1),
                "support": int(report_dict[lbl]["support"]),
            })

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred_l2, labels=class_labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
        ax=ax,
        linewidths=1.0,
        linecolor="#e2e8f0"
    )
    ax.set_title("Confusion Matrix - Logistic Regression (L2)", fontsize=13, fontweight="bold", color=TITLE_COLOR, pad=12)
    ax.set_xlabel("Predicted Rental Tier", fontsize=11, color=LABEL_COLOR, labelpad=8)
    ax.set_ylabel("Actual Rental Tier", fontsize=11, color=LABEL_COLOR, labelpad=8)
    plt.tight_layout()
    plt.savefig(_chart_path("logistic_regression_confusion_matrix.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")

    # Save best model to disk for live predictor use
    with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(model_l2, f)

    results = {
        "models": {
            "none": {
                "name": "Logistic Regression (No Regularization)",
                "penalty": "None",
                "description": "Unconstrained cross-entropy loss optimization.",
                **metrics_unreg
            },
            "l2": {
                "name": "Logistic Regression (L2 Regularized)",
                "penalty": "L2 (Ridge)",
                "description": "Quadratic parameter penalty (C=1.0), best out-of-sample generalization.",
                **metrics_l2
            },
            "l1": {
                "name": "Logistic Regression (L1 Regularized)",
                "penalty": "L1 (Lasso)",
                "description": "Laplacian parameter penalty via SAGA solver inducing feature sparsity.",
                **metrics_l1
            }
        },
        "per_class": per_class_metrics,
        "class_labels": class_labels,
        "thresholds": thresholds,
        "total_test": len(y_test),
        "charts": ["logistic_regression_confusion_matrix.png"]
    }

    with open(CACHE_METRICS_PATH, "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    print("Testing logistic_regression.py...")
    res = run_logistic_regression(force_retrain=True)
    print("L2 Test Acc:", res["models"]["l2"]["test_accuracy"])
    print("L1 Test Acc:", res["models"]["l1"]["test_accuracy"])
    print("Per-class metrics:", res["per_class"])