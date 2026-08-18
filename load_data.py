import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

DATA_FILES = [
    os.path.join(DATA_DIR, "apartments_for_rent_classified_10K.xlsx"),
    os.path.join(DATA_DIR, "apartments_for_rent_classified_100K.xlsx"),
]


def load_data() -> pd.DataFrame:
    """Load both Excel datasets and combine them without changing their values."""
    frames = []

    for path in DATA_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_excel(path)
        df["_dataset_source"] = os.path.basename(path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def get_data_summary() -> dict:
    df = load_data()

    summary = {
        "n_rows": df.shape[0],
        "n_cols": len([c for c in df.columns if c != "_dataset_source"]),
        "files": [os.path.basename(p) for p in DATA_FILES],
        "file_rows": {},
        "columns": [c for c in df.columns if c != "_dataset_source"],
        "dtypes": {},
        "missing_counts": {},
        "preview": df.drop(columns=["_dataset_source"]).head(10).to_dict("records"),
    }

    for path in DATA_FILES:
        name = os.path.basename(path)
        summary["file_rows"][name] = int((df["_dataset_source"] == name).sum())

    for col in summary["columns"]:
        summary["dtypes"][col] = str(df[col].dtype)
        summary["missing_counts"][col] = int(df[col].isnull().sum())

    return summary


if __name__ == "__main__":
    print(get_data_summary())
