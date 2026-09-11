import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import preprocess_data, get_feature_names

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_METRICS_PATH = os.path.join(MODELS_DIR, "linreg_metrics.pkl")

SERIES_COLOR = "#5ba0d7"
TITLE_COLOR = "#1f3a5f"
LABEL_COLOR = "#4a5568"
GRID_COLOR = "#e2e8f0"


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _apply_theme(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=TITLE_COLOR, pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=LABEL_COLOR, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, color=LABEL_COLOR, labelpad=8)
    ax.grid(axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cbd5e1")
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors=LABEL_COLOR, labelsize=9.5)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")


def calculate_metrics(y_true, y_pred, y_train=None, y_train_pred=None) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    test_r2 = r2_score(y_true, y_pred)
    train_r2 = r2_score(y_train, y_train_pred) if y_train is not None else test_r2

    # Mean Absolute Percentage Error (MAPE) avoiding division by zero
    valid_mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100)

    return {
        "train_r2": round(float(train_r2), 4),
        "test_r2": round(float(test_r2), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
    }


def run_linear_regression(force_retrain: bool = False) -> dict:
    """
    Train and evaluate:
    1. Linear Regression (Without Regularization / OLS)
    2. Ridge Regression (L2 Regularization)
    3. Lasso Regression (L1 Regularization)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    if not force_retrain and os.path.exists(CACHE_METRICS_PATH):
        try:
            with open(CACHE_METRICS_PATH, "rb") as f:
                cached_data = pickle.load(f)
            # Ensure plots exist on disk
            if os.path.exists(_chart_path("actual_vs_predicted.png")) and os.path.exists(_chart_path("residuals.png")):
                return cached_data
        except Exception:
            pass

    print("[LinearRegression] Preprocessing dataset...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

    print("[LinearRegression] Training Models...")
    # 1. Ordinary Least Squares (Without Regularization)
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols_test = ols.predict(X_test)
    y_pred_ols_train = ols.predict(X_train)
    metrics_ols = calculate_metrics(y_test, y_pred_ols_test, y_train, y_pred_ols_train)

    # 2. Ridge (L2)
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge_test = ridge.predict(X_test)
    y_pred_ridge_train = ridge.predict(X_train)
    metrics_ridge = calculate_metrics(y_test, y_pred_ridge_test, y_train, y_pred_ridge_train)

    # 3. Lasso (L1)
    lasso = Lasso(alpha=1.0, max_iter=2000, random_state=42)
    lasso.fit(X_train, y_train)
    y_pred_lasso_test = lasso.predict(X_test)
    y_pred_lasso_train = lasso.predict(X_train)
    metrics_lasso = calculate_metrics(y_test, y_pred_lasso_test, y_train, y_pred_lasso_train)

    # Zero coefficients count for Lasso (feature selection demonstration)
    lasso_zero_weights = int(np.sum(np.abs(lasso.coef_) < 1e-5))

    # Top Features Analysis from Ridge/OLS
    try:
        feature_names = get_feature_names(preprocessor)
        coef_series = pd.Series(ridge.coef_, index=feature_names)
        top_positive = coef_series.sort_values(ascending=False).head(8)
        top_negative = coef_series.sort_values(ascending=True).head(8)

        top_drivers = [
            {"feature": feat.replace("num__", "").replace("cat__", "").replace("remainder__", ""),
             "coefficient": round(float(coef), 2),
             "impact": "Positive" if coef > 0 else "Negative"}
            for feat, coef in top_positive.items()
        ]
    except Exception as e:
        print(f"Warning extracting feature names: {e}")
        top_drivers = []

    # Visualizations: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    # Subsample for plot clarity
    sample_indices = np.random.RandomState(42).choice(len(y_test), size=min(1500, len(y_test)), replace=False)
    y_test_sample = np.array(y_test)[sample_indices]
    y_pred_sample = np.array(y_pred_ols_test)[sample_indices]

    ax.scatter(y_test_sample, y_pred_sample, alpha=0.35, color=SERIES_COLOR, edgecolors="none", s=25, label="Test Listings")
    min_val = min(y_test_sample.min(), y_pred_sample.min(), 0)
    max_val = max(y_test_sample.max(), y_pred_sample.max(), 5000)
    ax.plot([min_val, max_val], [min_val, max_val], color=TITLE_COLOR, linestyle="--", linewidth=2, label="Ideal 45° Fit Line")
    ax.set_xlim(0, min(max_val, 6000))
    ax.set_ylim(0, min(max_val, 6000))
    _apply_theme(ax, "Actual vs Predicted Rental Price (Linear Regression)", "Actual Rent ($ USD)", "Predicted Rent ($ USD)")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    _save("actual_vs_predicted.png")

    # Visualizations: Residuals Distribution
    residuals = y_test - y_pred_ols_test
    # Clip residuals to +/- $2000 for meaningful inspection
    res_clipped = residuals[(residuals > -2500) & (residuals < 2500)]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.hist(res_clipped, bins=45, color=SERIES_COLOR, edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#b91c1c", linestyle="--", linewidth=2, label="Zero Error Reference")
    _apply_theme(ax, "Residual Errors Distribution (Actual - Predicted)", "Residual Error ($ USD)", "Frequency")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    _save("residuals.png")

    # Save models to disk for live predictor use
    with open(os.path.join(MODELS_DIR, "linear_regression.pkl"), "wb") as f:
        pickle.dump(ols, f)
    with open(os.path.join(MODELS_DIR, "ridge_regression.pkl"), "wb") as f:
        pickle.dump(ridge, f)
    with open(os.path.join(MODELS_DIR, "lasso_regression.pkl"), "wb") as f:
        pickle.dump(lasso, f)

    results = {
        "models": {
            "ols": {
                "name": "Linear Regression (OLS)",
                "regularization": "None",
                "description": "Unregularized Ordinary Least Squares minimizing Mean Squared Error.",
                **metrics_ols
            },
            "ridge": {
                "name": "Ridge Regression",
                "regularization": "L2 (Weight Decay)",
                "description": "Penalizes sum of squared coefficients (alpha=10.0), stabilizing multicollinear predictors.",
                **metrics_ridge
            },
            "lasso": {
                "name": "Lasso Regression",
                "regularization": "L1 (Feature Sparsity)",
                "description": f"Penalizes absolute sum of weights (alpha=1.0), zeroing out {lasso_zero_weights} redundant features.",
                **metrics_lasso
            }
        },
        "top_drivers": top_drivers,
        "lasso_zero_features": lasso_zero_weights,
        "test_samples": len(y_test),
        "train_samples": len(y_train),
        "charts": ["actual_vs_predicted.png", "residuals.png"]
    }

    with open(CACHE_METRICS_PATH, "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    print("Testing linear_regression.py...")
    res = run_linear_regression(force_retrain=True)
    print("OLS Metrics:", res["models"]["ols"])
    print("Ridge Metrics:", res["models"]["ridge"])
    print("Lasso Metrics:", res["models"]["lasso"])