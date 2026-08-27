from flask import Flask, render_template, request

from load_data import get_data_summary
from apartment_eda import run_eda
from preprocessing import preprocess_data
from linear_regression import run_linear_regression
from logistic_regression import run_logistic_regression


app = Flask(__name__)


# =========================================================
# HOME / DASHBOARD
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none",
        summary=None,
        error=None
    )


# =========================================================
# DATA LOADING
# =========================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        # Page number
        page = request.args.get(
            "page",
            default=1,
            type=int
        )

        # Rows per page
        page_size = request.args.get(
            "page_size",
            default=20,
            type=int
        )

        # Allowed page sizes
        if page_size not in [10, 20, 50, 100]:
            page_size = 20

        # Prevent invalid page numbers
        if page < 1:
            page = 1

        # Load dataset summary
        summary = get_data_summary(
            page=page,
            page_size=page_size
        )

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
def eda():

    error = None
    eda_output = None

    try:

        eda_output = run_eda()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error
    )


# =========================================================
# PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing():

    error = None
    preprocessing_output = None

    try:

        (
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            preprocessor
        ) = preprocess_data()

        preprocessing_output = {

            "original_rows":
                len(y_train) + len(y_test),

            "training_samples":
                len(y_train),

            "testing_samples":
                len(y_test),

            "original_features":
                13,

            "processed_features":
                X_train_processed.shape[1],

            "train_shape":
                X_train_processed.shape,

            "test_shape":
                X_test_processed.shape,

            "target_name":
                "price"
        }

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        results=preprocessing_output,
        error=error
    )


# =========================================================
# LINEAR REGRESSION
# =========================================================

@app.route("/linear-regression")
def linear_regression():

    error = None
    results = None

    try:

        results = run_linear_regression()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        results=results,
        error=error
    )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

@app.route("/logistic-regression")
def logistic_regression():

    error = None
    results = None

    try:

        results = run_logistic_regression()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        results=results,
        error=error
    )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )