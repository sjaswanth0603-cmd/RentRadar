import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from load_data import load_data

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close("all")


def _clean_old_charts():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    for filename in os.listdir(CHARTS_DIR):
        if filename.endswith(".png"):
            try:
                os.remove(os.path.join(CHARTS_DIR, filename))
            except OSError:
                pass


def run_eda():
    data = load_data()

    original_columns = [
        c for c in data.columns if c != "_dataset_source"
    ]

    _clean_old_charts()
    charts = []

    # Missing values
    missing = data[original_columns].isnull().sum()
    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values("missing_count", ascending=False)

    if not missing_df.empty:
        plt.figure(figsize=(11, 5))
        plt.bar(
            range(len(missing_df)),
            missing_df["missing_pct"].values
        )
        plt.xticks(
            range(len(missing_df)),
            missing_df.index,
            rotation=55,
            ha="right"
        )
        plt.ylabel("Missing (%)")
        plt.title("Missing Values by Column")
        _save("missing_values.png")
        charts.append("missing_values.png")

    # Duplicates
    duplicate_count = int(
        data[original_columns].duplicated().sum()
    )

    duplicate_id_count = 0

    if "id" in data.columns:
        duplicate_id_count = int(
            data["id"].duplicated().sum()
        )

    # Price
    if "price" in data.columns:
        price = pd.to_numeric(
            data["price"],
            errors="coerce"
        ).dropna()

        if not price.empty:
            plt.figure(figsize=(9, 5))
            plt.hist(price, bins=40)
            plt.title("Rental Price Distribution")
            plt.xlabel("Price")
            plt.ylabel("Count")
            _save("price_distribution.png")
            charts.append("price_distribution.png")

            plt.figure(figsize=(9, 4))
            plt.boxplot(price, vert=False)
            plt.title("Rental Price - Boxplot")
            plt.xlabel("Price")
            _save("price_boxplot.png")
            charts.append("price_boxplot.png")

    # Square feet
    if "square_feet" in data.columns:
        sqft = pd.to_numeric(
            data["square_feet"],
            errors="coerce"
        ).dropna()

        if not sqft.empty:
            plt.figure(figsize=(9, 5))
            plt.hist(sqft, bins=40)
            plt.title("Square Feet Distribution")
            plt.xlabel("Square Feet")
            plt.ylabel("Count")
            _save("square_feet_distribution.png")
            charts.append("square_feet_distribution.png")

    # Bedrooms
    if "bedrooms" in data.columns:
        bedrooms = pd.to_numeric(
            data["bedrooms"],
            errors="coerce"
        ).dropna()

        if not bedrooms.empty:
            counts = bedrooms.value_counts().sort_index()

            plt.figure(figsize=(9, 5))
            plt.bar(
                counts.index.astype(str),
                counts.values
            )
            plt.title("Apartment Count by Bedrooms")
            plt.xlabel("Bedrooms")
            plt.ylabel("Count")
            _save("bedrooms_distribution.png")
            charts.append("bedrooms_distribution.png")

    # Bathrooms
    if "bathrooms" in data.columns:
        bathrooms = pd.to_numeric(
            data["bathrooms"],
            errors="coerce"
        ).dropna()

        if not bathrooms.empty:
            plt.figure(figsize=(9, 5))
            plt.hist(bathrooms, bins=20)
            plt.title("Bathroom Distribution")
            plt.xlabel("Bathrooms")
            plt.ylabel("Count")
            _save("bathrooms_distribution.png")
            charts.append("bathrooms_distribution.png")

    # Top cities
    if "cityname" in data.columns:
        top_cities = (
            data["cityname"]
            .fillna("Unknown")
            .value_counts()
            .head(15)
        )

        plt.figure(figsize=(11, 6))
        plt.barh(
            range(len(top_cities)),
            top_cities.values
        )
        plt.yticks(
            range(len(top_cities)),
            top_cities.index
        )
        plt.gca().invert_yaxis()
        plt.title("Top 15 Cities by Number of Listings")
        plt.xlabel("Listings")
        _save("top_cities.png")
        charts.append("top_cities.png")

    # Top states
    if "state" in data.columns:
        top_states = (
            data["state"]
            .fillna("Unknown")
            .value_counts()
            .head(15)
        )

        plt.figure(figsize=(10, 6))
        plt.barh(
            range(len(top_states)),
            top_states.values
        )
        plt.yticks(
            range(len(top_states)),
            top_states.index
        )
        plt.gca().invert_yaxis()
        plt.title("Top 15 States by Number of Listings")
        plt.xlabel("Listings")
        _save("top_states.png")
        charts.append("top_states.png")

    # Photo availability
    if "has_photo" in data.columns:
        photo_counts = (
            data["has_photo"]
            .fillna("Unknown")
            .value_counts()
        )

        plt.figure(figsize=(8, 5))
        plt.bar(
            photo_counts.index.astype(str),
            photo_counts.values
        )
        plt.title("Listings by Photo Availability")
        plt.xlabel("Has Photo")
        plt.ylabel("Count")
        _save("photo_availability.png")
        charts.append("photo_availability.png")

    # Pets allowed
    if "pets_allowed" in data.columns:
        pets = (
            data["pets_allowed"]
            .fillna("Not Specified")
            .value_counts()
            .head(10)
        )

        plt.figure(figsize=(9, 5))
        plt.bar(
            pets.index.astype(str),
            pets.values
        )
        plt.xticks(rotation=30, ha="right")
        plt.title("Pets Allowed")
        plt.xlabel("Pets Allowed")
        plt.ylabel("Listings")
        _save("pets_allowed.png")
        charts.append("pets_allowed.png")

    # Price vs Square Feet
    if (
        "price" in data.columns
        and "square_feet" in data.columns
    ):
        plot_df = data[
            ["price", "square_feet"]
        ].copy()

        plot_df["price"] = pd.to_numeric(
            plot_df["price"],
            errors="coerce"
        )

        plot_df["square_feet"] = pd.to_numeric(
            plot_df["square_feet"],
            errors="coerce"
        )

        plot_df = plot_df.dropna()

        plot_df = plot_df[
            (plot_df["price"] >= 0)
            & (plot_df["square_feet"] > 0)
        ].head(10000)

        if not plot_df.empty:
            plt.figure(figsize=(9, 5))
            plt.scatter(
                plot_df["square_feet"],
                plot_df["price"],
                alpha=0.25,
                s=10
            )
            plt.title("Price vs Square Feet")
            plt.xlabel("Square Feet")
            plt.ylabel("Price")
            _save("price_vs_square_feet.png")
            charts.append("price_vs_square_feet.png")

    # Correlation heatmap
    numeric = data[
        original_columns
    ].select_dtypes(include="number")

    if numeric.shape[1] >= 2:
        corr = numeric.corr()

        plt.figure(figsize=(10, 7))
        plt.imshow(
            corr,
            aspect="auto",
            interpolation="nearest"
        )

        plt.colorbar(label="Correlation")

        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=60,
            ha="right"
        )

        plt.yticks(
            range(len(corr.columns)),
            corr.columns
        )

        plt.title("Numeric Feature Correlation Heatmap")
        _save("correlation_heatmap.png")
        charts.append("correlation_heatmap.png")

    # Missing values dictionary
    missing_dict = {
        str(col): int(cnt)
        for col, cnt in missing.items()
        if cnt > 0
    }

    # Numeric summary
    numeric_summary = {}

    for col in [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms"
    ]:
        if col in data.columns:
            series = pd.to_numeric(
                data[col],
                errors="coerce"
            ).dropna()

            if not series.empty:
                numeric_summary[col] = {
                    "min": round(float(series.min()), 2),
                    "max": round(float(series.max()), 2),
                    "mean": round(float(series.mean()), 2),
                    "median": round(float(series.median()), 2)
                }

    # Dataset sources
    if "_dataset_source" in data.columns:
        source_counts = (
            data["_dataset_source"]
            .value_counts()
            .to_dict()
        )
    else:
        source_counts = {}

    return {
        "n_rows": len(data),
        "n_cols": len(original_columns),
        "duplicate_count": duplicate_count,
        "duplicate_id_count": duplicate_id_count,
        "missing": missing_dict,
        "source_counts": {
            str(k): int(v)
            for k, v in source_counts.items()
        },
        "numeric_summary": numeric_summary,
        "charts": charts
    }


if __name__ == "__main__":
    print(run_eda())