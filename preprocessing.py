import os
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from load_data import load_data

# =========================================================
# CACHED PREPROCESSING ARTIFACTS
# =========================================================
_CACHED_PREPROCESSED: Optional[Tuple[Any, Any, pd.Series, pd.Series, ColumnTransformer]] = None
_CACHED_SUMMARY: Optional[Dict[str, Any]] = None
_CACHED_CLASSIFICATION: Optional[Tuple[Any, Any, pd.Series, pd.Series, ColumnTransformer, Dict[str, float]]] = None

# =========================================================
# EXCLUDED COLUMNS & ANTI-LEAKAGE RATIONALE
# =========================================================
EXCLUDED_COLUMNS_RATIONALE = {
    "id": "Arbitrary database primary key with no generalizable relationship to rent. Overfits if retained.",
    "price_display": "TARGET LEAKAGE: Direct text representation of rental price (e.g. '$1,450'). Artificially inflates metrics.",
    "currency": "Constant feature: 100% of listings are USD ($). Zero variance provides zero predictive information.",
    "address": "Extremely high-cardinality street addresses with heavy missingness (~60%). Geographic coordinates (lat/long) and city/state are used instead.",
    "time": "Crawl/scrape timestamp epoch: collection metadata with no bearing on property valuation.",
    "title": "Raw unstructured marketing headlines: parsed into engineered binary luxury/renovation flags and description length.",
    "body": "Long-form description text: parsed into engineered feature flags and text length metrics.",
    "amenities": "Raw comma-separated strings: parsed into 13 distinct binary amenity indicators and total amenity count."
}

# Top 13 amenities extracted into tabular predictors
AMENITY_PATTERNS = {
    "has_parking": ["parking", "garage"],
    "has_pool": ["pool"],
    "has_gym": ["gym", "fitness"],
    "has_washer_dryer": ["washer dryer", "washer/dryer", "laundry"],
    "has_ac": ["ac", "air condition", "central air"],
    "has_dishwasher": ["dishwasher"],
    "has_patio_deck": ["patio", "deck", "balcony"],
    "has_storage": ["storage"],
    "has_clubhouse": ["clubhouse"],
    "has_fireplace": ["fireplace"],
    "has_wood_floors": ["wood floor", "hardwood"],
    "has_gated": ["gated"],
    "has_elevator": ["elevator"]
}

LUXURY_KEYWORDS = ["luxury", "renovated", "remodeled", "granite", "stainless", "upscale", "penthouse", "modern"]


# =========================================================
# FEATURE ENGINEERING: EXTRACT ALL AMENITIES & RATIOS
# =========================================================
def extract_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw amenities, text descriptions, and structural dimensions into
    clean numeric features:
    1. 13 Binary amenity flags (Pool, Gym, Parking, In-Unit Laundry, AC, Dishwasher, etc.)
    2. Total amenity count
    3. Luxury/renovated finish flag from title/body text
    4. Description length
    5. Bed-to-bath ratio & square-feet per bedroom ratio
    """
    df = df.copy()

    # 1. Parse amenities column
    amenities_str = df["amenities"].fillna("").astype(str).str.lower()
    for feat_name, patterns in AMENITY_PATTERNS.items():
        pattern_regex = "|".join(patterns)
        df[feat_name] = amenities_str.str.contains(pattern_regex, regex=True).astype(int)

    # Total amenity count
    df["amenity_count"] = amenities_str.apply(lambda s: len([t for t in s.split(",") if t.strip()]) if s else 0)

    # 2. Text NLP flags from title and body
    combined_text = (df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)).str.lower()
    luxury_regex = "|".join(LUXURY_KEYWORDS)
    df["feat_luxury"] = combined_text.str.contains(luxury_regex, regex=True).astype(int)
    df["desc_length"] = df["body"].fillna("").astype(str).str.len().clip(upper=2000)

    # 3. Structural ratios (safe division with small epsilon)
    beds = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(1.0).clip(lower=0)
    baths = pd.to_numeric(df["bathrooms"], errors="coerce").fillna(1.0).clip(lower=0.5)
    sqft = pd.to_numeric(df["square_feet"], errors="coerce").fillna(850.0).clip(lower=150)

    df["bed_bath_ratio"] = (beds / (baths + 0.1)).round(2)
    df["sqft_per_bed"] = (sqft / (beds + 0.5)).round(1)

    return df


# =========================================================
# REGRESSION PREPROCESSING PIPELINE
# =========================================================
def preprocess_data(force_reload: bool = False) -> Tuple[Any, Any, pd.Series, pd.Series, ColumnTransformer]:
    """
    Main regression preprocessing pipeline:
    1. Loads deduplicated rental data
    2. Performs comprehensive feature engineering (amenities, ratios, luxury flags)
    3. Validates target (price > 0 and not null)
    4. Drops raw text, identifiers, and leakage fields
    5. Splits into reproducible 80% train / 20% test (random_state=42)
    6. Fits ColumnTransformer strictly on training set:
       - Numerical: Median imputation + StandardScaler
       - Categorical: Most frequent imputation + OneHotEncoder(handle_unknown='ignore')
    7. Returns transformed matrices and fitted pipeline.
    """
    global _CACHED_PREPROCESSED, _CACHED_SUMMARY

    if _CACHED_PREPROCESSED is not None and not force_reload:
        return _CACHED_PREPROCESSED

    print("\n========== PREPROCESSING STARTED ==========")
    data = load_data()
    original_shape = data.shape

    if "_dataset_source" in data.columns:
        data = data.drop(columns=["_dataset_source"])

    dup_before = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    # Target column validation
    target = "price"
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found.")

    # Numerical candidates conversion
    base_num = ["price", "bathrooms", "bedrooms", "square_feet", "latitude", "longitude"]
    for col in base_num:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Target cleaning
    missing_target = int(data[target].isna().sum())
    invalid_target = int((data[target] <= 0).sum())
    data = data[data[target].notna() & (data[target] > 0)].copy()

    # Feature Engineering across all available fields
    print("Extracting all engineered features & amenities...")
    data = extract_engineered_features(data)

    # Exclude raw text, IDs, and leakage fields
    cols_to_drop = [c for c in EXCLUDED_COLUMNS_RATIONALE.keys() if c in data.columns]
    data = data.drop(columns=cols_to_drop)

    # Separate X and y
    X = data.drop(columns=[target])
    y = data[target]

    numerical_columns = list(X.select_dtypes(include=["number"]).columns)
    categorical_columns = list(X.select_dtypes(include=["object", "category", "string"]).columns)

    # Missing value report
    missing_report = {}
    for col in X.columns:
        cnt = int(X[col].isnull().sum())
        if cnt > 0:
            missing_report[col] = {
                "count": cnt,
                "pct": round((cnt / len(X)) * 100, 2)
            }

    # 80/20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ]
    )

    # Fit strictly on training split
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Store comprehensive summary for dashboard
    _CACHED_SUMMARY = {
        "original_rows": original_shape[0],
        "original_cols": original_shape[1],
        "cleaned_rows": len(data),
        "target_missing": missing_target,
        "target_invalid": invalid_target,
        "duplicate_rows": dup_before,
        "dropped_columns": cols_to_drop,
        "dropped_reasons": EXCLUDED_COLUMNS_RATIONALE,
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "training_samples": len(y_train),
        "testing_samples": len(y_test),
        "processed_features": X_train_processed.shape[1],
        "missing_report": missing_report,
        "target_name": "price",
        "engineered_amenities": list(AMENITY_PATTERNS.keys()) + ["amenity_count"],
        "engineered_structural": ["bed_bath_ratio", "sqft_per_bed", "feat_luxury", "desc_length"]
    }

    _CACHED_PREPROCESSED = (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )

    print(f"Preprocessing completed: {X_train_processed.shape[1]} features "
          f"({len(numerical_columns)} numeric incl. amenities, {len(categorical_columns)} categorical)")
    return _CACHED_PREPROCESSED


# =========================================================
# FEATURE NAMES EXTRACTOR
# =========================================================
def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts human-readable feature names from the fitted ColumnTransformer."""
    feature_names = []
    try:
        raw_names = preprocessor.get_feature_names_out()
        for name in raw_names:
            cleaned = name.split("__", 1)[-1]
            feature_names.append(cleaned)
    except Exception:
        for name, pipe, cols in preprocessor.transformers_:
            if name == "numerical":
                feature_names.extend(cols)
            elif name == "categorical":
                try:
                    encoder = pipe.named_steps["encoder"]
                    ohe_names = encoder.get_feature_names_out(cols)
                    feature_names.extend(list(ohe_names))
                except Exception:
                    feature_names.extend(cols)
    return feature_names


# =========================================================
# PRICE TIER CLASSIFICATION PREPROCESSING
# =========================================================
def preprocess_classification_data(
    force_reload: bool = False
) -> Tuple[Any, Any, pd.Series, pd.Series, ColumnTransformer, Dict[str, float]]:
    """
    Derived Classification Pipeline:
    Creates balanced price tiers (Budget, Mid-Range, Premium) with full engineered
    amenities and strict zero price leakage.
    """
    global _CACHED_CLASSIFICATION

    if _CACHED_CLASSIFICATION is not None and not force_reload:
        return _CACHED_CLASSIFICATION

    data = load_data().copy()
    if "_dataset_source" in data.columns:
        data = data.drop(columns=["_dataset_source"])

    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data[data["price"].notna() & (data["price"] > 0)].copy()

    # Extract all features
    data = extract_engineered_features(data)

    # Derive price_tier quantiles from population
    q33 = float(data["price"].quantile(0.333))
    q66 = float(data["price"].quantile(0.666))

    def categorize_tier(p: float) -> str:
        if p <= q33:
            return "Budget"
        elif p <= q66:
            return "Mid-Range"
        else:
            return "Premium"

    data["price_tier"] = data["price"].apply(categorize_tier)

    # Drop non-predictive & leakage fields (PRICE IS STRICTLY EXCLUDED)
    cols_to_drop = [c for c in EXCLUDED_COLUMNS_RATIONALE.keys() if c in data.columns]
    data = data.drop(columns=cols_to_drop)

    X = data.drop(columns=["price", "price_tier"])
    y = data["price_tier"]

    numerical_columns = list(X.select_dtypes(include=["number"]).columns)
    categorical_columns = list(X.select_dtypes(include=["object", "category", "string"]).columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", num_pipe, numerical_columns),
            ("categorical", cat_pipe, categorical_columns)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    thresholds = {
        "budget_max": round(q33, 2),
        "mid_range_max": round(q66, 2),
        "class_distribution": {
            "Budget": int((y == "Budget").sum()),
            "Mid-Range": int((y == "Mid-Range").sum()),
            "Premium": int((y == "Premium").sum())
        }
    }

    _CACHED_CLASSIFICATION = (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        thresholds
    )

    return _CACHED_CLASSIFICATION


# =========================================================
# SUMMARY ACCESSOR FOR FLASK
# =========================================================
def get_preprocessing_summary() -> Dict[str, Any]:
    """Returns the cached preprocessing summary dictionary for display in templates/preprocessing.html."""
    global _CACHED_SUMMARY
    if _CACHED_SUMMARY is None:
        preprocess_data()
    return _CACHED_SUMMARY or {}


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, prep = preprocess_data(force_reload=True)
    summary = get_preprocessing_summary()
    print("Features extracted successfully:", len(get_feature_names(prep)))
    print("Engineered amenities:", summary["engineered_amenities"])
    print("Engineered structural:", summary["engineered_structural"])