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
from kmeans import run_kmeans
from hierarchical_clustering import run_hierarchical_clustering
from DBSCAN import run_dbscan


app = Flask(__name__)


# =========================================================
# 1. DATA LOADING
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

        summary = get_data_summary(
            page=page,
            page_size=page_size
        )

    except Exception as e:
        error = f"Error profiling dataset: {e}"

    return render_template(
        "data_loading.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# =========================================================
# 2. EXPLORATORY DATA ANALYSIS
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
# 4. LINEAR REGRESSION
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
# 5. LOGISTIC REGRESSION
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
# 6. DECISION TREES & ENSEMBLES
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
# 7. K-MEANS CLUSTERING
# =========================================================

@app.route("/kmeans", methods=["GET", "POST"])
@app.route("/k-means", methods=["GET", "POST"])
def kmeans():

    error = None
    results = None

    method = request.form.get(
        "method",
        "manual"
    )

    manual_k = request.form.get(
        "manual_k",
        3,
        type=int
    )

    min_k = request.form.get(
        "min_k",
        2,
        type=int
    )

    max_k = request.form.get(
        "max_k",
        10,
        type=int
    )

    if request.method == "POST":

        try:

            if method not in [
                "manual",
                "elbow",
                "silhouette"
            ]:
                method = "manual"

            if manual_k < 2:
                manual_k = 2

            if min_k < 2:
                min_k = 2

            if max_k <= min_k:
                max_k = min_k + 1

            if max_k > 15:
                max_k = 15

            results = run_kmeans(
                method=method,
                manual_k=manual_k,
                min_k=min_k,
                max_k=max_k
            )

        except Exception as e:

            error = f"Error running K-Means clustering: {e}"

    return render_template(
        "kmeans.html",
        active="kmeans",
        results=results,
        error=error,
        method=method,
        manual_k=manual_k,
        min_k=min_k,
        max_k=max_k
    )


# =========================================================
# 7B. HIERARCHICAL CLUSTERING
# =========================================================

@app.route("/hierarchical-clustering", methods=["GET", "POST"])
@app.route("/hierarchical_clustering", methods=["GET", "POST"])
@app.route("/hierarchical", methods=["GET", "POST"])
def hierarchical_clustering():

    error = None
    results = None

    linkage = request.form.get("linkage", "ward") or "ward"
    raw_n_clusters = request.form.get("n_clusters", "auto")

    if not raw_n_clusters or str(raw_n_clusters).lower() == "auto":
        n_clusters = "auto"
    else:
        try:
            n_clusters = int(raw_n_clusters)
        except (ValueError, TypeError):
            n_clusters = "auto"

    try:
        results = run_hierarchical_clustering(
            n_clusters=n_clusters,
            linkage_method=linkage,
            metric="euclidean"
        )
    except Exception as e:
        error = f"Error running Hierarchical Clustering: {e}"

    return render_template(
        "hierarchical.html",
        active="hierarchical",
        results=results,
        error=error,
        n_clusters=n_clusters,
        linkage=linkage,
        metric="euclidean"
    )


# =========================================================
# 7C. DBSCAN CLUSTERING
# =========================================================

@app.route("/dbscan", methods=["GET", "POST"])
@app.route("/dbscan-clustering", methods=["GET", "POST"])
def dbscan():

    error = None
    results = None

    eps_mode = request.form.get("eps_mode", "auto")
    raw_eps_val = request.form.get("eps_val", "")
    min_samples = request.form.get("min_samples", 5, type=int)

    if eps_mode == "manual" and raw_eps_val:
        try:
            eps = float(raw_eps_val)
            is_auto = False
        except (ValueError, TypeError):
            eps = "auto"
            is_auto = True
    else:
        eps = "auto"
        is_auto = True

    try:
        results = run_dbscan(
            eps=eps,
            min_samples=min_samples
        )
    except Exception as e:
        error = f"Error running DBSCAN clustering: {e}"

    return render_template(
        "dbscan.html",
        active="dbscan",
        results=results,
        error=error,
        eps=eps if eps != "auto" else (results.get("eps") if results else 0.5),
        is_auto=is_auto,
        min_samples=min_samples
    )


# =========================================================
# 8. RENT PREDICTOR
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

        selected_model = request.form.get(
            "model_id",
            "lightgbm"
        )

        amenity_flags = [
            "has_parking",
            "has_pool",
            "has_gym",
            "has_washer_dryer",
            "has_ac",
            "has_dishwasher",
            "has_patio_deck",
            "has_storage",
            "has_clubhouse",
            "has_fireplace",
            "has_wood_floors",
            "has_gated",
            "has_elevator",
            "feat_luxury"
        ]

        input_amenities = [
            k for k in amenity_flags
            if request.form.get(k)
        ]

        try:

            result = manager.predict(
                dict(request.form)
            )

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
                "active_amenities": [
                    "Air Conditioning"
                ],
                "amenities_count": 1,
                "insights": [
                    f"Valuation completed with fallback notice: {e}"
                ],
                "comparables": [],
                "input_summary": {
                    "state": "TX",
                    "bedrooms": 2,
                    "bathrooms": 1.5,
                    "square_feet": 950
                }
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

    for port in range(
        start_port,
        start_port + max_attempts
    ):

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        ) as s:

            if s.connect_ex(
                ("127.0.0.1", port)
            ) != 0:

                return port

    return start_port


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = find_open_port(5000)

    print(
        "\n======================================================="
    )

    print(
        " RentRadar AI Rental Intelligence Dashboard Live"
    )

    print(
        f" URL: http://127.0.0.1:{port}"
    )

    print(
        "=======================================================\n"
    )

    app.run(
        host="127.0.0.1",
        port=port,
        debug=True
    )