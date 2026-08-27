import os
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "datasets"
)


DATA_FILES = [
    os.path.join(
        DATA_DIR,
        "apartments_for_rent_classified_10K.xlsx"
    ),

    os.path.join(
        DATA_DIR,
        "apartments_for_rent_classified_100K.xlsx"
    ),
]


# =========================================================
# LOAD DATA
# =========================================================

def load_data() -> pd.DataFrame:

    """
    Load both Excel datasets and combine them.

    The original values are not modified.
    """

    frames = []

    for path in DATA_FILES:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        df = pd.read_excel(path)

        # Keep track of source internally
        df["_dataset_source"] = os.path.basename(path)

        frames.append(df)

    combined_df = pd.concat(
        frames,
        ignore_index=True
    )

    return combined_df


# =========================================================
# DATA LOADING SUMMARY
# =========================================================

def get_data_summary(
    page=1,
    page_size=20
):

    df = load_data()

    # Remove internal column when displaying data
    display_df = df.drop(
        columns=["_dataset_source"],
        errors="ignore"
    )

    # -----------------------------------------------------
    # BASIC INFORMATION
    # -----------------------------------------------------

    total_rows = display_df.shape[0]

    total_columns = display_df.shape[1]

    total_cells = (
        total_rows *
        total_columns
    )

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------

    duplicate_rows = int(
        display_df.duplicated().sum()
    )

    # -----------------------------------------------------
    # MEMORY USAGE
    # -----------------------------------------------------

    memory_bytes = display_df.memory_usage(
        deep=True
    ).sum()

    memory_kb = memory_bytes / 1024

    memory_mb = memory_kb / 1024

    # -----------------------------------------------------
    # DATASET FILES
    # -----------------------------------------------------

    files = []

    for path in DATA_FILES:

        name = os.path.basename(path)

        file_df = df[
            df["_dataset_source"] == name
        ]

        files.append({

            "name": name,

            "rows": int(
                len(file_df)
            )

        })

    # -----------------------------------------------------
    # PAGINATION
    # -----------------------------------------------------

    total_pages = max(
        1,
        (total_rows + page_size - 1)
        // page_size
    )

    # Prevent invalid page numbers
    page = max(
        1,
        min(page, total_pages)
    )

    start = (
        page - 1
    ) * page_size

    end = start + page_size

    page_df = display_df.iloc[
        start:end
    ]

    # Convert NaN values to empty strings
    # only for displaying the table
    page_df = page_df.where(
        pd.notnull(page_df),
        ""
    )

    records = page_df.to_dict(
        orient="records"
    )

    # -----------------------------------------------------
    # RETURN SUMMARY
    # -----------------------------------------------------

    return {

        "total_rows":
            total_rows,

        "total_columns":
            total_columns,

        "total_cells":
            total_cells,

        "duplicate_rows":
            duplicate_rows,

        "memory_usage":
            f"{memory_mb:.2f} MB",

        "files":
            files,

        "columns":
            list(display_df.columns),

        "records":
            records,

        "page":
            page,

        "page_size":
            page_size,

        "total_pages":
            total_pages,

        "has_previous":
            page > 1,

        "has_next":
            page < total_pages

    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    summary = get_data_summary()

    print(
        "Total Rows:",
        summary["total_rows"]
    )

    print(
        "Total Columns:",
        summary["total_columns"]
    )

    print(
        "Total Cells:",
        summary["total_cells"]
    )

    print(
        "Duplicate Rows:",
        summary["duplicate_rows"]
    )

    print(
        "Memory Usage:",
        summary["memory_usage"]
    )