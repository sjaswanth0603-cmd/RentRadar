import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from load_data import load_data


def preprocess_data():

    print("\n========== PREPROCESSING STARTED ==========")

    # =========================================================
    # 1. LOAD DATA
    # =========================================================

    data = load_data()

    print("\n========== ORIGINAL DATASET ==========")
    print("Shape:", data.shape)

    print("\nColumns:")
    print(list(data.columns))


    # =========================================================
    # 2. REMOVE INTERNAL DATASET SOURCE COLUMN
    # =========================================================

    if "_dataset_source" in data.columns:

        data = data.drop(columns=["_dataset_source"])

        print("\nRemoved internal column: _dataset_source")


    # =========================================================
    # 3. DUPLICATE HANDLING
    # =========================================================

    print("\n========== DUPLICATE HANDLING ==========")

    duplicate_before = int(data.duplicated().sum())

    print(
        "Duplicate rows before removal:",
        duplicate_before
    )

    data = data.drop_duplicates().copy()

    duplicate_after = int(data.duplicated().sum())

    print(
        "Duplicate rows after removal:",
        duplicate_after
    )


    # =========================================================
    # 4. TARGET VARIABLE
    # =========================================================

    print("\n========== TARGET VARIABLE ==========")

    target = "price"

    if target not in data.columns:

        raise ValueError(
            "Target column 'price' was not found."
        )

    print("Target column:", target)


    # =========================================================
    # 5. NUMERICAL CONVERSION
    # =========================================================

    print("\n========== NUMERICAL CONVERSION ==========")

    numerical_candidates = [
        "price",
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude"
    ]

    for col in numerical_candidates:

        if col in data.columns:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

            print(
                f"{col}: converted to numeric"
            )


    # =========================================================
    # 6. TARGET CLEANING
    # =========================================================

    print("\n========== TARGET CLEANING ==========")

    target_missing_before = int(
        data[target].isna().sum()
    )

    invalid_price_count = int(
        (data[target] <= 0).sum()
    )

    print(
        "Missing price values:",
        target_missing_before
    )

    print(
        "Invalid price values (<= 0):",
        invalid_price_count
    )


    # Remove rows where target is missing
    data = data[
        data[target].notna()
    ].copy()


    # Remove rows where price is zero/negative
    data = data[
        data[target] > 0
    ].copy()


    print(
        "Dataset shape after target cleaning:",
        data.shape
    )


    # =========================================================
    # 7. REMOVE UNNECESSARY / LEAKAGE COLUMNS
    # =========================================================

    print("\n========== REMOVING UNNECESSARY COLUMNS ==========")

    columns_to_remove = [

        # Identifier
        "id",

        # Free-text fields
        "title",
        "body",
        "amenities",

        # Almost constant / unnecessary
        "currency",

        # TARGET LEAKAGE
        "price_display",

        # Extremely high-cardinality and heavily missing
        "address",

        # Raw timestamp
        "time"
    ]


    removed_columns = []

    for col in columns_to_remove:

        if col in data.columns:

            removed_columns.append(col)

            data = data.drop(
                columns=[col]
            )


    print("Removed columns:")

    for col in removed_columns:

        print(" -", col)


    # =========================================================
    # 8. SEPARATE FEATURES AND TARGET
    # =========================================================

    print("\n========== FEATURE / TARGET SEPARATION ==========")

    X = data.drop(
        columns=[target]
    )

    y = data[target]

    print("X shape:", X.shape)
    print("y shape:", y.shape)


    # =========================================================
    # 9. IDENTIFY NUMERICAL FEATURES
    # =========================================================

    numerical_columns = list(
        X.select_dtypes(
            include=["number"]
        ).columns
    )


    # =========================================================
    # 10. IDENTIFY CATEGORICAL FEATURES
    # =========================================================

    categorical_columns = list(
        X.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    )


    print("\n========== FEATURE TYPES ==========")

    print("\nNumerical features:")

    for col in numerical_columns:

        print(" -", col)


    print("\nCategorical features:")

    for col in categorical_columns:

        print(" -", col)


    # =========================================================
    # 11. MISSING VALUE REPORT
    # =========================================================

    print("\n========== MISSING VALUES ==========")

    missing = X.isnull().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )


    if missing.empty:

        print("No missing values.")

    else:

        for col, count in missing.items():

            percentage = (
                count / len(X)
            ) * 100

            print(
                f"{col}: "
                f"{count} missing "
                f"({percentage:.2f}%)"
            )


    # =========================================================
    # 12. TRAIN / TEST SPLIT
    # =========================================================

    print("\n========== TRAIN / TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )


    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    # =========================================================
    # 13. NUMERICAL PIPELINE
    # =========================================================

    numerical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",

                StandardScaler()
            )

        ]
    )


    # =========================================================
    # 14. CATEGORICAL PIPELINE
    # =========================================================

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

                    handle_unknown="ignore",

                    sparse_output=False
                )
            )

        ]
    )


    # =========================================================
    # 15. COMBINE PREPROCESSING
    # =========================================================

    print("\n========== BUILDING PREPROCESSOR ==========")

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",

                numerical_pipeline,

                numerical_columns
            ),

            (
                "categorical",

                categorical_pipeline,

                categorical_columns
            )

        ]
    )


    # =========================================================
    # 16. FIT ON TRAINING DATA
    # =========================================================

    print(
        "\nFitting preprocessing on training data..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )


    # =========================================================
    # 17. TRANSFORM TEST DATA
    # =========================================================

    print(
        "Transforming testing data..."
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )


    # =========================================================
    # 18. FINAL RESULTS
    # =========================================================

    print(
        "\n========== PREPROCESSING COMPLETED =========="
    )

    print(
        "\nOriginal dataset:",
        data.shape
    )

    print(
        "\nBefore preprocessing:"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )


    print(
        "\nAfter preprocessing:"
    )

    print(
        "X_train_processed:",
        X_train_processed.shape
    )

    print(
        "X_test_processed :",
        X_test_processed.shape
    )


    print(
        "\ny_train:",
        y_train.shape
    )

    print(
        "y_test:",
        y_test.shape
    )


    print(
        "\n========== PREPROCESSING SUCCESSFUL =========="
    )


    # =========================================================
    # 19. RETURN EVERYTHING NEEDED FOR ML
    # =========================================================

    return (

        X_train_processed,

        X_test_processed,

        y_train,

        y_test,

        preprocessor

    )


# =============================================================
# RUN DIRECTLY
# =============================================================

if __name__ == "__main__":

    preprocess_data()