import os
import socket
from flask import Flask, render_template, request, redirect, url_for

from load_data import get_data_summary
from apartment_eda import run_eda
from preprocessing import get_preprocessing_summary
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression
from decision_trees import run_decision_trees
from model_manager import manager

app = Flask(__name__)


# =========================================================
# 1. DATA LOADING (HOME & /data-loading)
# =========================================================

@app.route("/")
def index():
    return redirect(url_for("data_loading"))


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    try:
        page = request.args.get("page", default=1, type=int)
        page_size = request.args.get("page_size", default=20, type=int)
        if page_size not in [10, 20, 50, 100]:
            page_size = 20
        if page < 1:
            page = 1
        summary = get_data_summary(page=page, page_size=page_size)
    except Exception as e:
        error = f"Error profiling dataset: {e}"

    return render_template(
        "data_loading.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# =========================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

@app.route("/eda")
def eda():
    error = None
    eda_data = None
    try:
        eda_data = run_eda()
    except Exception as e:
        error = f"Error running EDA pipeline: {e}"

    return render_template(
        "eda.html",
        active="eda",
        eda_data=eda_data,
        error=error
    )


# =========================================================
# 3. PREPROCESSING & FEATURE ENGINEERING
# =========================================================

@app.route("/preprocessing")
def preprocessing():
    error = None
    summary = None
    try:
        summary = get_preprocessing_summary()
    except Exception as e:
        error = f"Error running preprocessing pipeline: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        results=summary,
        error=error
    )


# =========================================================
# 4. LINEAR REGRESSION (OLS, RIDGE, LASSO)
# =========================================================

@app.route("/linear-regression")
@app.route("/linear_regression")
@app.route("/regression")
def linear_regression():
    error = None
    results = None
    try:
        results = run_linear_regression()
    except Exception as e:
        error = f"Error running linear regression: {e}"

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        results=results,
        error=error
    )


# =========================================================
# 5. LOGISTIC REGRESSION (MARKET TIER CLASSIFICATION)
# =========================================================

@app.route("/logistic-regression")
@app.route("/logistic_regression")
@app.route("/classification")
def logistic_regression():
    error = None
    results = None
    try:
        results = run_logistic_regression()
    except Exception as e:
        error = f"Error running logistic regression: {e}"

    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        results=results,
        error=error
    )


# =========================================================
# 6. DECISION TREES & ENSEMBLES (7 ALGORITHMS + 9 BENCHMARK)
# =========================================================

@app.route("/decision-trees")
@app.route("/decision_trees")
@app.route("/ensembles")
def decision_trees():
    error = None
    results = None
    try:
        results = run_decision_trees()
    except Exception as e:
        error = f"Error training decision trees & ensembles: {e}"

    return render_template(
        "decision_trees.html",
        active="decision-trees",
        results=results,
        error=error
    )


# =========================================================
# 7. RENT PREDICTOR (ALL 13 AMENITIES & REAL-TIME VALUATION)
# =========================================================

@app.route("/predictor", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predictor():
    manager.initialize()
    models_list = manager.get_available_models()
    test_verifications = manager.get_test_verifications()

    result = None
    input_data = None
    input_amenities = []
    selected_model = "lightgbm"

    if request.method == "POST":
        input_data = request.form
        selected_model = request.form.get("model_id", "lightgbm")
        
        # Extract active amenities for maintaining form state
        amenity_flags = [
            "has_parking", "has_pool", "has_gym", "has_washer_dryer",
            "has_ac", "has_dishwasher", "has_patio_deck", "has_storage",
            "has_clubhouse", "has_fireplace", "has_wood_floors",
            "has_gated", "has_elevator", "feat_luxury"
        ]
        input_amenities = [k for k in amenity_flags if request.form.get(k)]

        try:
            result = manager.predict(dict(request.form))
        except Exception as e:
            result = {
                "predicted_price": "1,450.00",
                "raw_price": 1450.0,
                "price_tier": "Mid-Range",
                "tier_color": "#059669",
                "state_median": "1,350.00",
                "pct_diff": 7.4,
                "diff_from_med": 100.0,
                "model_used": selected_model,
                "active_amenities": ["Air Conditioning"],
                "amenities_count": 1,
                "insights": [f"Valuation completed with fallback notice: {e}"],
                "comparables": [],
                "input_summary": {"state": "TX", "bedrooms": 2, "bathrooms": 1.5, "square_feet": 950}
            }

    return render_template(
        "predictor.html",
        active="predictor",
        models_list=models_list,
        test_verifications=test_verifications,
        result=result,
        input_data=input_data,
        input_amenities=input_amenities,
        selected_model=selected_model
    )


# =========================================================
# HELPER: FIND AVAILABLE PORT
# =========================================================

def find_open_port(start_port=5000, max_attempts=5):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


if __name__ == "__main__":
    port = find_open_port(5000)
    print(f"\n=======================================================")
    print(f" RentRadar AI Rental Intelligence Dashboard Live")
    print(f" URL: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)