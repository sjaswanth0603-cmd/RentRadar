import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing import preprocess_data


# =========================================================
# LINEAR REGRESSION
# =========================================================

def run_linear_regression():

    print("\n=================================================")
    print("              LINEAR REGRESSION")
    print("=================================================")

    # -----------------------------------------------------
    # 1. PREPROCESS DATA
    # -----------------------------------------------------

    (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data()

    print("\nPreprocessing completed.")

    print(
        "Training shape:",
        X_train_processed.shape
    )

    print(
        "Testing shape:",
        X_test_processed.shape
    )

    # -----------------------------------------------------
    # 2. TRAIN MODEL
    # -----------------------------------------------------

    print("\nTraining Linear Regression...")

    model = LinearRegression()

    model.fit(
        X_train_processed,
        y_train
    )

    print("Linear Regression training completed.")

    # -----------------------------------------------------
    # 3. PREDICT
    # -----------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(
        X_test_processed
    )

    # -----------------------------------------------------
    # 4. METRICS
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\n========== RESULTS ==========")

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"MSE  : {mse:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    # -----------------------------------------------------
    # 5. SAVE GRAPH
    # -----------------------------------------------------

    charts_dir = os.path.join(
        os.path.dirname(__file__),
        "static",
        "charts"
    )

    os.makedirs(
        charts_dir,
        exist_ok=True
    )

    chart_path = os.path.join(
        charts_dir,
        "linear_regression_actual_vs_predicted.png"
    )

    print("\nCreating graph...")

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.35
    )

    minimum = min(
        y_test.min(),
        y_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linewidth=2
    )

    plt.xlabel(
        "Actual Rent Price"
    )

    plt.ylabel(
        "Predicted Rent Price"
    )

    plt.title(
        "Linear Regression - Actual vs Predicted Rent"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        chart_path,
        dpi=120
    )

    plt.close()

    print(
        "Graph saved:",
        chart_path
    )

    # -----------------------------------------------------
    # 6. SAMPLE RESULTS
    # -----------------------------------------------------

    y_test_values = y_test.reset_index(
        drop=True
    )

    samples = []

    for i in range(
        min(10, len(y_test_values))
    ):

        samples.append({

            "actual":
                round(
                    float(y_test_values.iloc[i]),
                    2
                ),

            "predicted":
                round(
                    float(y_pred[i]),
                    2
                )

        })

    # -----------------------------------------------------
    # 7. RETURN RESULTS
    # -----------------------------------------------------

    return {

        "model_name":
            "Linear Regression",

        "target":
            "price",

        "training_samples":
            len(y_train),

        "testing_samples":
            len(y_test),

        "features":
            X_train_processed.shape[1],

        "mae":
            round(mae, 4),

        "mse":
            round(mse, 4),

        "rmse":
            round(rmse, 4),

        "r2":
            round(r2, 4),

        "r2_percentage":
            round(r2 * 100, 2),

        "chart":
            "charts/linear_regression_actual_vs_predicted.png",

        "samples":
            samples
    }


# =========================================================
# TEST DIRECTLY
# =========================================================

if __name__ == "__main__":

    results = run_linear_regression()

    print("\n========== FINAL RESULTS ==========")

    print(results)