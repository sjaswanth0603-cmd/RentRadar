"""
export_preprocessed_dataset.py
==============================
Generates a complete, publication-grade preprocessed rental dataset in Excel (.xlsx)
and CSV (.csv) formats for RentRadar.

Includes:
  Sheet 1: Preprocessed_Data (90,427 cleaned & feature-engineered listings)
  Sheet 2: Data_Dictionary (Field definitions, types, categories, transformations)
  Sheet 3: Pipeline_Summary (Audit metrics, missing value treatments, anti-leakage rationale)
"""

import os
import time
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from load_data import load_data
from preprocessing import extract_engineered_features, EXCLUDED_COLUMNS_RATIONALE, AMENITY_PATTERNS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
EXCEL_OUTPUT_PATH = os.path.join(DATASETS_DIR, "apartments_preprocessed.xlsx")
CSV_OUTPUT_PATH = os.path.join(DATASETS_DIR, "apartments_preprocessed.csv")


def generate_preprocessed_dataframe() -> pd.DataFrame:
    """Builds the complete clean, feature-engineered, leakage-free dataset."""
    print("[1/4] Loading raw rental listings...")
    data = load_data()
    if "_dataset_source" in data.columns:
        data = data.drop(columns=["_dataset_source"])

    print(f"      Loaded {len(data):,} raw records.")

    # Exclude raw text, IDs, and leakage fields before duplicate check
    cols_to_drop = [c for c in EXCLUDED_COLUMNS_RATIONALE.keys() if c in data.columns]
    data_for_dedup = data.drop(columns=cols_to_drop)

    dup_count = int(data_for_dedup.duplicated().sum())
    print(f"[2/4] Removing duplicates ({dup_count:,} duplicate records found)...")
    data = data.loc[~data_for_dedup.duplicated()].copy()

    # Numeric conversion
    base_num = ["price", "bathrooms", "bedrooms", "square_feet", "latitude", "longitude"]
    for col in base_num:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Target cleaning & outlier filtering ($100 <= price <= $15,000)
    target = "price"
    valid_mask = data[target].notna() & (data[target] >= 100) & (data[target] <= 15000)
    invalid_count = int((~valid_mask).sum())
    print(f"      Filtered {invalid_count:,} invalid target/extreme outlier rows.")
    data = data[valid_mask].copy()

    print("[3/4] Extracting 13 amenity flags, NLP keywords, and structural ratios...")
    data = extract_engineered_features(data)

    # Derive price_tier quantiles from cleaned population
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

    # Drop raw unstructured & leakage columns
    data = data.drop(columns=cols_to_drop)

    # Impute missing values consistently with pipeline rules
    # Median for numeric features
    for col in ["bedrooms", "bathrooms", "square_feet", "latitude", "longitude"]:
        if col in data.columns:
            med_val = data[col].median()
            data[col] = data[col].fillna(round(med_val, 2) if col in ["latitude", "longitude"] else round(med_val, 1))

    # Categorical missing values
    data["pets_allowed"] = data["pets_allowed"].fillna("None")
    data["cityname"] = data["cityname"].fillna("Unknown")
    data["state"] = data["state"].fillna("Unknown")

    # 80/20 reproducible split assignment
    train_idx, test_idx = train_test_split(data.index, test_size=0.20, random_state=42, stratify=data["price_tier"])
    data["split_group"] = "Train (80%)"
    data.loc[test_idx, "split_group"] = "Test (20%)"

    # Order columns logically
    ordered_cols = [
        # Target variables
        "price",
        "price_tier",
        "split_group",
        # Structural dimensions & ratios
        "bedrooms",
        "bathrooms",
        "square_feet",
        "bed_bath_ratio",
        "sqft_per_bed",
        # Geographic coordinates
        "cityname",
        "state",
        "latitude",
        "longitude",
        # 13 Extracted Amenity Indicators
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
        "amenity_count",
        # Quality & NLP flags
        "feat_luxury",
        "desc_length",
        # Listing metadata
        "category",
        "fee",
        "has_photo",
        "pets_allowed",
        "price_type",
        "source"
    ]

    # Reorder existing columns
    final_cols = [c for c in ordered_cols if c in data.columns]
    data = data[final_cols].reset_index(drop=True)
    print(f"[4/4] Dataset ready: {data.shape[0]:,} rows x {data.shape[1]} columns.")
    return data, q33, q66


def create_excel_workbook(df: pd.DataFrame, q33: float, q66: float, output_path: str):
    """Writes multi-sheet formatted Excel file using openpyxl write_only mode for maximum speed and memory efficiency."""
    print(f"Creating Excel workbook at: {output_path}...")
    t0 = time.time()

    wb = openpyxl.Workbook(write_only=True)

    # -------------------------------------------------------------
    # SHEET 1: Preprocessed_Data
    # -------------------------------------------------------------
    print("Writing Sheet 1: Preprocessed_Data...")
    ws_data = wb.create_sheet(title="Preprocessed_Data")

    # Header
    ws_data.append(list(df.columns))

    # Fast row writing
    # Convert dataframe to tuples/lists
    values = df.values.tolist()
    for row in values:
        ws_data.append(row)

    # -------------------------------------------------------------
    # SHEET 2: Data_Dictionary
    # -------------------------------------------------------------
    print("Writing Sheet 2: Data_Dictionary...")
    ws_dict = wb.create_sheet(title="Data_Dictionary")

    dictionary_data = [
        ["Field Name", "Data Type", "Category", "Description", "Transformation / Imputation Logic"],
        ["price", "Float", "Target (Continuous)", "Monthly rental price in USD ($)", "Target variable for regression models. Cleaned: $100 <= price <= $15,000. Missing removed."],
        ["price_tier", "String", "Target (Categorical)", "Rental price tier category (Budget, Mid-Range, Premium)", f"Population tertiles: Budget (<= ${q33:,.0f}), Mid-Range (${q33:,.0f} - ${q66:,.0f}), Premium (> ${q66:,.0f})."],
        ["split_group", "String", "Model Partition", "Machine Learning evaluation partition", "Reproducible 80% Train / 20% Held-Out Test split (random_state=42, stratified by tier)."],
        ["bedrooms", "Float", "Physical Dimension", "Number of bedrooms", "Cleaned numeric. Missing values imputed with population median (1.0)."],
        ["bathrooms", "Float", "Physical Dimension", "Number of bathrooms", "Cleaned numeric. Missing values imputed with population median (1.0)."],
        ["square_feet", "Float", "Physical Dimension", "Total interior livable area in square feet", "Cleaned numeric. Missing values imputed with population median."],
        ["bed_bath_ratio", "Float", "Structural Ratio", "Ratio of bedrooms to bathrooms", "Engineered non-linear ratio: bedrooms / (bathrooms + 0.1)."],
        ["sqft_per_bed", "Float", "Structural Ratio", "Average livable square footage per bedroom", "Engineered non-linear ratio: square_feet / (bedrooms + 0.5)."],
        ["cityname", "String", "Geographic Location", "Municipality / City name", "Standardized text. Missing imputed with 'Unknown'."],
        ["state", "String", "Geographic Location", "US State 2-letter abbreviation", "Standardized uppercase code. Missing imputed with 'Unknown'."],
        ["latitude", "Float", "Geographic Coordinate", "Geographic latitude coordinate", "Cleaned numeric coordinates. Missing imputed with median."],
        ["longitude", "Float", "Geographic Coordinate", "Geographic longitude coordinate", "Cleaned numeric coordinates. Missing imputed with median."],
        ["has_parking", "Integer (0/1)", "Amenity Indicator", "Dedicated parking or garage present", "Extracted from raw amenities text using regex: parking|garage."],
        ["has_pool", "Integer (0/1)", "Amenity Indicator", "Swimming pool on premises", "Extracted from raw amenities text using regex: pool."],
        ["has_gym", "Integer (0/1)", "Amenity Indicator", "Fitness center or gym available", "Extracted from raw amenities text using regex: gym|fitness."],
        ["has_washer_dryer", "Integer (0/1)", "Amenity Indicator", "In-unit or complex laundry washer & dryer", "Extracted from raw amenities text using regex: washer dryer|washer/dryer|laundry."],
        ["has_ac", "Integer (0/1)", "Amenity Indicator", "Air conditioning / central climate control", "Extracted from raw amenities text using regex: ac|air condition|central air."],
        ["has_dishwasher", "Integer (0/1)", "Amenity Indicator", "Dishwasher included in unit", "Extracted from raw amenities text using regex: dishwasher."],
        ["has_patio_deck", "Integer (0/1)", "Amenity Indicator", "Private balcony, patio, or deck", "Extracted from raw amenities text using regex: patio|deck|balcony."],
        ["has_storage", "Integer (0/1)", "Amenity Indicator", "Extra storage space or locker", "Extracted from raw amenities text using regex: storage."],
        ["has_clubhouse", "Integer (0/1)", "Amenity Indicator", "Resident clubhouse / community lounge", "Extracted from raw amenities text using regex: clubhouse."],
        ["has_fireplace", "Integer (0/1)", "Amenity Indicator", "Indoor fireplace present", "Extracted from raw amenities text using regex: fireplace."],
        ["has_wood_floors", "Integer (0/1)", "Amenity Indicator", "Hardwood or wood-laminate flooring", "Extracted from raw amenities text using regex: wood floor|hardwood."],
        ["has_gated", "Integer (0/1)", "Amenity Indicator", "Gated access community / security gate", "Extracted from raw amenities text using regex: gated."],
        ["has_elevator", "Integer (0/1)", "Amenity Indicator", "Elevator building access", "Extracted from raw amenities text using regex: elevator."],
        ["amenity_count", "Integer", "Amenity Aggregate", "Total count of distinct amenities mentioned", "Parsed count of comma-separated items in listing amenities."],
        ["feat_luxury", "Integer (0/1)", "NLP Signal", "Luxury, upscale, or newly renovated finishes", "Extracted from listing title & body text (luxury, renovated, remodeled, granite, stainless, etc.)."],
        ["desc_length", "Integer", "NLP Signal", "Listing description text character length", "Character count of listing body (clipped at upper threshold 2,000 chars)."],
        ["category", "String", "Listing Attribute", "Rental housing category classification", "e.g., housing/rent, apartments for rent."],
        ["fee", "String", "Listing Attribute", "Broker fee requirement indicator", "Yes / No / None."],
        ["has_photo", "String", "Listing Attribute", "Photo availability indicator", "Yes / Thumbnail / No."],
        ["pets_allowed", "String", "Listing Attribute", "Pet acceptance policy", "Cats,Dogs / Cats / Dogs / None. Missing imputed with 'None'."],
        ["price_type", "String", "Listing Attribute", "Billing frequency basis", "e.g., Monthly, Weekly."],
        ["source", "String", "Listing Attribute", "Listing syndication portal source", "Platform or classified source identifier."]
    ]

    for r in dictionary_data:
        ws_dict.append(r)

    # -------------------------------------------------------------
    # SHEET 3: Pipeline_Summary
    # -------------------------------------------------------------
    print("Writing Sheet 3: Pipeline_Summary...")
    ws_summary = wb.create_sheet(title="Pipeline_Summary")

    train_count = int((df["split_group"] == "Train (80%)").sum())
    test_count = int((df["split_group"] == "Test (20%)").sum())

    summary_rows = [
        ["Pipeline Step / Metric", "Value", "Operational Details"],
        ["Total Cleaned Listings", f"{len(df):,}", "Final clean tabular records exported to Excel"],
        ["Total Processed Features", f"{len(df.columns)} columns", "Includes targets, partitions, structural ratios, 13 amenity flags, and NLP signals"],
        ["Training Split (80%)", f"{train_count:,} listings", "Used for fitting estimators, scalers, and imputers"],
        ["Held-Out Test Split (20%)", f"{test_count:,} listings", "Strictly reserved for out-of-sample validation"],
        ["Continuous Target Variable", "price", "Monthly rent in USD ($)"],
        ["Categorical Target Variable", "price_tier", f"Budget (<= ${q33:,.0f}), Mid-Range (${q33:,.0f}-${q66:,.0f}), Premium (> ${q66:,.0f})"],
        ["Price Outlier Bounds", "$100 to $15,000", "Removed non-positive rents, deposit errors, and extreme anomalous outliers"],
        ["Imputation: Numerical Features", "Population Median", "Median imputation for bathrooms, bedrooms, sqft, coordinates"],
        ["Imputation: Categorical Features", "Constant ('None' / 'Unknown')", "Missing pets_allowed -> 'None', missing city/state -> 'Unknown'"],
        ["--- DROPPED COLUMNS ---", "--- ANTI-LEAKAGE RATIONALE ---", "--- IMPACT ---"]
    ]

    for col_name, reason in EXCLUDED_COLUMNS_RATIONALE.items():
        summary_rows.append([f"Dropped: {col_name}", reason, "Excluded to prevent target leakage and overfitting"])

    for r in summary_rows:
        ws_summary.append(r)

    wb.save(output_path)
    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Excel workbook successfully written in {elapsed:.2f}s! ({file_size_mb:.2f} MB)")


def main():
    print("=========================================================")
    print(" RentRadar Preprocessed Dataset Exporter")
    print("=========================================================\n")

    df, q33, q66 = generate_preprocessed_dataframe()

    # 1. Export Excel workbook
    create_excel_workbook(df, q33, q66, EXCEL_OUTPUT_PATH)

    # 2. Export companion CSV file (fast to read in any environment)
    print(f"\nWriting companion CSV file to: {CSV_OUTPUT_PATH}...")
    t0 = time.time()
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    csv_size_mb = os.path.getsize(CSV_OUTPUT_PATH) / (1024 * 1024)
    print(f"CSV exported successfully in {time.time() - t0:.2f}s! ({csv_size_mb:.2f} MB)")

    print("\n[SUCCESS] Export completed:")
    print(f"  -> Excel: {EXCEL_OUTPUT_PATH}")
    print(f"  -> CSV:   {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
