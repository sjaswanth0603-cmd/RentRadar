import os
import pickle
import pandas as pd
import numpy as np

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

DATA_FILES = [
    os.path.join(DATA_DIR, "apartments_for_rent_classified_10K.xlsx"),
    os.path.join(DATA_DIR, "apartments_for_rent_classified_100K.xlsx"),
]

CACHE_FILE = os.path.join(DATA_DIR, ".cache_dedup.pkl")


# =========================================================
# LOAD DATA WITH FAST DISK CACHING
# =========================================================

def load_data(force_reload: bool = False, deduplicate: bool = True) -> pd.DataFrame:
    """
    Load apartment datasets with high-speed local caching.
    Deduplicates on primary key 'id' to prevent duplicate bias.
    """
    if not force_reload and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                df = pickle.load(f)
            return df
        except Exception:
            pass

    frames = []

    for path in DATA_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        df_part = pd.read_excel(path)
        df_part["_dataset_source"] = os.path.basename(path)
        frames.append(df_part)

    combined_df = pd.concat(frames, ignore_index=True)

    if deduplicate and "id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    # Save to binary pickle cache for fast subsequent reads
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(combined_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Warning: could not write cache file: {e}")

    return combined_df


def load_raw_data() -> pd.DataFrame:
    """Load both datasets without deduplication (for raw audit)."""
    frames = []
    for path in DATA_FILES:
        if os.path.exists(path):
            df_part = pd.read_excel(path)
            df_part["_dataset_source"] = os.path.basename(path)
            frames.append(df_part)
    return pd.concat(frames, ignore_index=True)


# =========================================================
# DATA LOADING SUMMARY
# =========================================================

def get_data_summary(page: int = 1, page_size: int = 20) -> dict:
    """
    Return dataset statistics and paginated record preview.
    """
    df = load_data()

    # Drop internal tracking column for display
    display_df = df.drop(columns=["_dataset_source"], errors="ignore")

    total_rows = int(display_df.shape[0])
    total_columns = int(display_df.shape[1])
    total_cells = int(total_rows * total_columns)
    duplicate_rows = int(display_df.duplicated().sum())

    memory_bytes = display_df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)

    # File breakdown
    files = []
    for path in DATA_FILES:
        name = os.path.basename(path)
        if "_dataset_source" in df.columns:
            count = int((df["_dataset_source"] == name).sum())
        else:
            count = 0
        files.append({"name": name, "rows": count})

    # Pagination
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))

    start = (page - 1) * page_size
    end = start + page_size

    page_df = display_df.iloc[start:end].copy()
    page_df = page_df.where(pd.notnull(page_df), "")
    records = page_df.to_dict(orient="records")

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_cells": total_cells,
        "duplicate_rows": duplicate_rows,
        "memory_usage": f"{memory_mb:.2f} MB",
        "files": files,
        "columns": list(display_df.columns),
        "records": records,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_previous": page > 1,
        "has_next": page < total_pages,
    }


if __name__ == "__main__":
    print("Testing load_data...")
    summary = get_data_summary()
    print("Total Rows:", summary["total_rows"])
    print("Total Columns:", summary["total_columns"])
    print("Memory Usage:", summary["memory_usage"])