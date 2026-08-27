import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from load_data import load_data


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

def run_logistic_regression():

    print()
    print("=" * 60)
    print("              LOGISTIC REGRESSION")
    print("=" * 60)
    print()

    # =====================================================
    # LOAD DATA
    # =====================================================

    print("========== LOADING DATASET ==========")

    df = load_data()

    print(f"Original dataset shape: {df.shape}")

    # Remove internal column
    if "_dataset_source" in df.columns:

        df = df.drop(columns=["_dataset_source"])

        print("Removed internal column: _dataset_source")


    # =====================================================
    # DUPLICATES
    # =====================================================

    print()
    print("========== DUPLICATE HANDLING ==========")

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows before removal: {duplicate_count}")

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Duplicate rows after removal: {df.duplicated().sum()}")


    # =====================================================
    # TARGET VARIABLE
    # =====================================================

    print()
    print("========== TARGET VARIABLE ==========")

    target_column = "price"

    print(f"Target column: {target_column}")


    # =====================================================
    # NUMERICAL CONVERSION
    # =====================================================

    print()
    print("========== NUMERICAL CONVERSION ==========")

    numerical_conversion_columns = [
        "price",
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude"
    ]

    for column in numerical_conversion_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            print(f"{column}: converted to numeric")


    # =====================================================
    # TARGET CLEANING
    # =====================================================

    print()
    print("========== TARGET CLEANING ==========")

    missing_price = df[target_column].isnull().sum()

    invalid_price = (
        (df[target_column] <= 0)
        & df[target_column].notnull()
    ).sum()

    print(f"Missing price values: {missing_price}")
    print(f"Invalid price values (<= 0): {invalid_price}")

    df = df[
        df[target_column].notnull()
        & (df[target_column] > 0)
    ].copy()

    print(
        f"Dataset shape after target cleaning: {df.shape}"
    )


    # =====================================================
    # REMOVE UNNECESSARY COLUMNS
    # =====================================================

    print()
    print("========== REMOVING UNNECESSARY COLUMNS ==========")

    columns_to_remove = [
        "id",
        "title",
        "body",
        "amenities",
        "currency",
        "price_display",
        "address",
        "time"
    ]

    removed_columns = []

    for column in columns_to_remove:

        if column in df.columns:

            df = df.drop(columns=[column])

            removed_columns.append(column)

    print("Removed columns:")

    for column in removed_columns:

        print(f" - {column}")


    # =====================================================
    # CREATE CLASSIFICATION TARGET
    # =====================================================

    print()
    print("========== CREATING CLASSIFICATION TARGET ==========")

    print()
    print("Logistic Regression requires a categorical target.")
    print("RentRadar price is converted into two classes:")
    print()
    print("  0 = Affordable")
    print("  1 = Expensive")
    print()

    # Use the median price as the classification boundary
    price_threshold = df[target_column].median()

    print(
        f"Price classification threshold: "
        f"{price_threshold:.2f}"
    )

    df["price_class"] = (
        df[target_column] > price_threshold
    ).astype(int)

    print()
    print("Class distribution:")

    class_counts = df["price_class"].value_counts().sort_index()

    print(
        f"Affordable (0): "
        f"{class_counts.get(0, 0)}"
    )

    print(
        f"Expensive (1): "
        f"{class_counts.get(1, 0)}"
    )


    # =====================================================
    # FEATURE / TARGET SEPARATION
    # =====================================================

    print()
    print("========== FEATURE / TARGET SEPARATION ==========")

    # Remove original continuous price
    # because price directly defines the class.
    X = df.drop(
        columns=[
            "price",
            "price_class"
        ]
    )

    y = df["price_class"]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


    # =====================================================
    # FEATURE TYPES
    # =====================================================

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print()
    print("========== FEATURE TYPES ==========")

    print()
    print("Numerical features:")

    for column in numerical_features:

        print(f" - {column}")

    print()
    print("Categorical features:")

    for column in categorical_features:

        print(f" - {column}")


    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    print()
    print("========== TRAIN / TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"Training samples: {len(y_train)}"
    )

    print(
        f"Testing samples: {len(y_test)}"
    )


    # =====================================================
    # PREPROCESSING
    # =====================================================

    print()
    print("========== BUILDING PREPROCESSOR ==========")

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


    # =====================================================
    # LOGISTIC REGRESSION MODEL
    # =====================================================

    print()
    print("========== BUILDING LOGISTIC REGRESSION MODEL ==========")

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    solver="liblinear",
                    random_state=42
                )
            )
        ]
    )


    # =====================================================
    # TRAIN MODEL
    # =====================================================

    print()
    print("Training Logistic Regression...")

    model.fit(
        X_train,
        y_train
    )

    print("Logistic Regression training completed.")


    # =====================================================
    # PREDICTIONS
    # =====================================================

    print()
    print("Generating predictions...")

    y_pred = model.predict(X_test)


    # =====================================================
    # METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # =====================================================
    # RESULTS
    # =====================================================

    print()
    print("=" * 60)
    print("                 RESULTS")
    print("=" * 60)

    print()
    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Affordable",
                "Expensive"
            ],
            zero_division=0
        )
    )


    # =====================================================
    # CONFUSION MATRIX GRAPH
    # =====================================================

    print()
    print("Creating confusion matrix graph...")

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    charts_dir = os.path.join(
        base_dir,
        "static",
        "charts"
    )

    os.makedirs(
        charts_dir,
        exist_ok=True
    )

    chart_path = os.path.join(
        charts_dir,
        "logistic_regression_confusion_matrix.png"
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(cm)

    plt.title(
        "Logistic Regression - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.xticks(
        [0, 1],
        ["Affordable", "Expensive"]
    )

    plt.yticks(
        [0, 1],
        ["Affordable", "Expensive"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        chart_path,
        dpi=150
    )

    plt.close()

    print(
        f"Graph saved: {chart_path}"
    )


    # =====================================================
    # SAMPLE PREDICTIONS
    # =====================================================

    samples = []

    for actual, predicted in zip(
        y_test.iloc[:10],
        y_pred[:10]
    ):

        samples.append(
            {
                "actual":
                    "Expensive"
                    if actual == 1
                    else "Affordable",

                "predicted":
                    "Expensive"
                    if predicted == 1
                    else "Affordable"
            }
        )


    # =====================================================
    # FINAL RESULTS
    # =====================================================

    results = {

        "model_name":
            "Logistic Regression",

        "target":
            "price_class",

        "classification_rule":
            f"price > {price_threshold:.2f} = Expensive",

        "training_samples":
            len(y_train),

        "testing_samples":
            len(y_test),

        "features":
            X.shape[1],

        "accuracy":
            round(accuracy, 4),

        "accuracy_percentage":
            round(accuracy * 100, 2),

        "precision":
            round(precision, 4),

        "recall":
            round(recall, 4),

        "f1":
            round(f1, 4),

        "confusion_matrix":
            cm.tolist(),

        "chart":
            "charts/logistic_regression_confusion_matrix.png",

        "samples":
            samples
    }


    print()
    print("=" * 60)
    print("             FINAL RESULTS")
    print("=" * 60)

    print(results)

    return results


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    run_logistic_regression()