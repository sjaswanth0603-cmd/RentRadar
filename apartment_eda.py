import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_data


CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename):
    plt.tight_layout()
    plt.savefig(
        _chart_path(filename),
        bbox_inches="tight"
    )
    plt.close("all")


def _clean_old_charts():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    for filename in os.listdir(CHARTS_DIR):
        if filename.endswith(".png"):
            try:
                os.remove(
                    os.path.join(CHARTS_DIR, filename)
                )
            except OSError:
                pass


def run_eda():

    # =========================================================
    # LOAD DATA
    # =========================================================

    data = load_data()

    original_columns = [
        c for c in data.columns
        if c != "_dataset_source"
    ]

    _clean_old_charts()

    charts = []


    # =========================================================
    # 1. MISSING VALUES
    # =========================================================

    missing = data[original_columns].isnull().sum()

    missing_pct = (
        missing / len(data)
    ) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False
    )

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

        plt.ylabel("Missing Percentage (%)")
        plt.xlabel("Column")
        plt.title("Missing Values by Column")

        _save("missing_values.png")

        charts.append("missing_values.png")


    # =========================================================
    # 2. DUPLICATES
    # =========================================================

    duplicate_count = int(
        data[original_columns]
        .duplicated()
        .sum()
    )

    duplicate_id_count = 0

    if "id" in data.columns:

        duplicate_id_count = int(
            data["id"]
            .duplicated()
            .sum()
        )


    # =========================================================
    # 3. OUTLIER ANALYSIS USING IQR
    # =========================================================

    outlier_counts = {}
    outlier_limits = {}

    for col in [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms"
    ]:

        if col not in data.columns:
            continue

        values = pd.to_numeric(
            data[col],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        outliers = values[
            (values < lower) |
            (values > upper)
        ]

        outlier_counts[col] = int(
            len(outliers)
        )

        outlier_limits[col] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower": round(float(lower), 2),
            "upper": round(float(upper), 2)
        }


    # =========================================================
    # 4. PRICE ANALYSIS
    # =========================================================

    if "price" in data.columns:

        price = pd.to_numeric(
            data["price"],
            errors="coerce"
        ).dropna()

        if not price.empty:

            plt.figure(figsize=(9, 5))

            plt.hist(
                price,
                bins=40
            )

            plt.title(
                "Rental Price Distribution"
            )

            plt.xlabel(
                "Rental Price"
            )

            plt.ylabel(
                "Number of Listings"
            )

            _save(
                "price_distribution.png"
            )

            charts.append(
                "price_distribution.png"
            )


            plt.figure(figsize=(9, 4))

            plt.boxplot(
                price,
                vert=False
            )

            plt.title(
                "Rental Price - Boxplot"
            )

            plt.xlabel(
                "Rental Price"
            )

            _save(
                "price_boxplot.png"
            )

            charts.append(
                "price_boxplot.png"
            )


    # =========================================================
    # 5. SQUARE FEET ANALYSIS
    # =========================================================

    if "square_feet" in data.columns:

        sqft = pd.to_numeric(
            data["square_feet"],
            errors="coerce"
        ).dropna()

        if not sqft.empty:

            plt.figure(figsize=(9, 5))

            plt.hist(
                sqft,
                bins=40
            )

            plt.title(
                "Apartment Size Distribution"
            )

            plt.xlabel(
                "Square Feet"
            )

            plt.ylabel(
                "Number of Listings"
            )

            _save(
                "square_feet_distribution.png"
            )

            charts.append(
                "square_feet_distribution.png"
            )


            plt.figure(figsize=(9, 4))

            plt.boxplot(
                sqft,
                vert=False
            )

            plt.title(
                "Square Feet - Boxplot"
            )

            plt.xlabel(
                "Square Feet"
            )

            _save(
                "square_feet_boxplot.png"
            )

            charts.append(
                "square_feet_boxplot.png"
            )


    # =========================================================
    # 6. BEDROOM ANALYSIS
    # =========================================================

    if "bedrooms" in data.columns:

        bedrooms = pd.to_numeric(
            data["bedrooms"],
            errors="coerce"
        ).dropna()

        if not bedrooms.empty:

            counts = (
                bedrooms
                .value_counts()
                .sort_index()
            )

            plt.figure(figsize=(9, 5))

            plt.bar(
                counts.index.astype(str),
                counts.values
            )

            plt.title(
                "Apartment Count by Bedrooms"
            )

            plt.xlabel(
                "Number of Bedrooms"
            )

            plt.ylabel(
                "Number of Listings"
            )

            _save(
                "bedrooms_distribution.png"
            )

            charts.append(
                "bedrooms_distribution.png"
            )


            plt.figure(figsize=(9, 4))

            plt.boxplot(
                bedrooms,
                vert=False
            )

            plt.title(
                "Bedrooms - Boxplot"
            )

            plt.xlabel(
                "Number of Bedrooms"
            )

            _save(
                "bedrooms_boxplot.png"
            )

            charts.append(
                "bedrooms_boxplot.png"
            )


    # =========================================================
    # 7. BATHROOM ANALYSIS
    # =========================================================

    if "bathrooms" in data.columns:

        bathrooms = pd.to_numeric(
            data["bathrooms"],
            errors="coerce"
        ).dropna()

        if not bathrooms.empty:

            counts = (
                bathrooms
                .value_counts()
                .sort_index()
            )

            plt.figure(figsize=(9, 5))

            plt.bar(
                counts.index.astype(str),
                counts.values
            )

            plt.title(
                "Apartment Count by Bathrooms"
            )

            plt.xlabel(
                "Number of Bathrooms"
            )

            plt.ylabel(
                "Number of Listings"
            )

            _save(
                "bathrooms_distribution.png"
            )

            charts.append(
                "bathrooms_distribution.png"
            )


            plt.figure(figsize=(9, 4))

            plt.boxplot(
                bathrooms,
                vert=False
            )

            plt.title(
                "Bathrooms - Boxplot"
            )

            plt.xlabel(
                "Number of Bathrooms"
            )

            _save(
                "bathrooms_boxplot.png"
            )

            charts.append(
                "bathrooms_boxplot.png"
            )


    # =========================================================
    # 8. TOP CITIES
    # =========================================================

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

        plt.title(
            "Top 15 Cities by Number of Listings"
        )

        plt.xlabel(
            "Number of Listings"
        )

        _save(
            "top_cities.png"
        )

        charts.append(
            "top_cities.png"
        )


    # =========================================================
    # 9. TOP STATES
    # =========================================================

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

        plt.title(
            "Top 15 States by Number of Listings"
        )

        plt.xlabel(
            "Number of Listings"
        )

        _save(
            "top_states.png"
        )

        charts.append(
            "top_states.png"
        )


    # =========================================================
    # 10. PHOTO AVAILABILITY
    # =========================================================

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

        plt.title(
            "Listings by Photo Availability"
        )

        plt.xlabel(
            "Photo Availability"
        )

        plt.ylabel(
            "Number of Listings"
        )

        _save(
            "photo_availability.png"
        )

        charts.append(
            "photo_availability.png"
        )


    # =========================================================
    # 11. PET POLICY
    # =========================================================

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

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.title(
            "Pet Policy Distribution"
        )

        plt.xlabel(
            "Pet Policy"
        )

        plt.ylabel(
            "Number of Listings"
        )

        _save(
            "pets_allowed.png"
        )

        charts.append(
            "pets_allowed.png"
        )


    # =========================================================
    # 12. PRICE VS SQUARE FEET
    # =========================================================

    if (
        "price" in data.columns
        and "square_feet" in data.columns
    ):

        plot_df = data[
            [
                "price",
                "square_feet"
            ]
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
        ]

        if len(plot_df) > 10000:

            plot_df = plot_df.sample(
                n=10000,
                random_state=42
            )

        if not plot_df.empty:

            plt.figure(figsize=(9, 5))

            plt.scatter(
                plot_df["square_feet"],
                plot_df["price"],
                alpha=0.25,
                s=10
            )

            plt.title(
                "Rental Price vs Apartment Size"
            )

            plt.xlabel(
                "Square Feet"
            )

            plt.ylabel(
                "Rental Price"
            )

            _save(
                "price_vs_square_feet.png"
            )

            charts.append(
                "price_vs_square_feet.png"
            )


    # =========================================================
    # 13. PRICE VS BEDROOMS
    # =========================================================

    if (
        "price" in data.columns
        and "bedrooms" in data.columns
    ):

        plot_df = data[
            [
                "price",
                "bedrooms"
            ]
        ].copy()

        plot_df["price"] = pd.to_numeric(
            plot_df["price"],
            errors="coerce"
        )

        plot_df["bedrooms"] = pd.to_numeric(
            plot_df["bedrooms"],
            errors="coerce"
        )

        plot_df = plot_df.dropna()

        plot_df = plot_df[
            (plot_df["price"] >= 0)
            & (plot_df["bedrooms"] >= 0)
        ]

        if len(plot_df) > 10000:

            plot_df = plot_df.sample(
                n=10000,
                random_state=42
            )

        if not plot_df.empty:

            plt.figure(figsize=(9, 5))

            plt.scatter(
                plot_df["bedrooms"],
                plot_df["price"],
                alpha=0.25,
                s=10
            )

            plt.title(
                "Rental Price vs Number of Bedrooms"
            )

            plt.xlabel(
                "Bedrooms"
            )

            plt.ylabel(
                "Rental Price"
            )

            _save(
                "price_vs_bedrooms.png"
            )

            charts.append(
                "price_vs_bedrooms.png"
            )


    # =========================================================
    # 14. PRICE VS BATHROOMS
    # =========================================================

    if (
        "price" in data.columns
        and "bathrooms" in data.columns
    ):

        plot_df = data[
            [
                "price",
                "bathrooms"
            ]
        ].copy()

        plot_df["price"] = pd.to_numeric(
            plot_df["price"],
            errors="coerce"
        )

        plot_df["bathrooms"] = pd.to_numeric(
            plot_df["bathrooms"],
            errors="coerce"
        )

        plot_df = plot_df.dropna()

        plot_df = plot_df[
            (plot_df["price"] >= 0)
            & (plot_df["bathrooms"] >= 0)
        ]

        if len(plot_df) > 10000:

            plot_df = plot_df.sample(
                n=10000,
                random_state=42
            )

        if not plot_df.empty:

            plt.figure(figsize=(9, 5))

            plt.scatter(
                plot_df["bathrooms"],
                plot_df["price"],
                alpha=0.25,
                s=10
            )

            plt.title(
                "Rental Price vs Number of Bathrooms"
            )

            plt.xlabel(
                "Bathrooms"
            )

            plt.ylabel(
                "Rental Price"
            )

            _save(
                "price_vs_bathrooms.png"
            )

            charts.append(
                "price_vs_bathrooms.png"
            )


    # =========================================================
    # 15. CORRELATION HEATMAP
    # =========================================================
    # =========================================================
    # CORRELATION HEATMAP
    # =========================================================

    correlation_columns = [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms",
        "latitude",
        "longitude"
    ]

    available_columns = [
        col
        for col in correlation_columns
        if col in data.columns
    ]

    if len(available_columns) >= 2:

        numeric = data[
            available_columns
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Calculate correlation
        corr = numeric.corr()

        # Create heatmap
        plt.figure(figsize=(10, 7))

        image = plt.imshow(
            corr,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )

        # Add correlation values inside cells
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                value = corr.iloc[i, j]

                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10
                )

        # Color bar
        plt.colorbar(
            image,
            label="Correlation"
        )

        # X-axis
        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=45,
            ha="right"
        )

        # Y-axis
        plt.yticks(
            range(len(corr.columns)),
            corr.columns
        )

        plt.title(
            "Correlation Heatmap of Apartment Features"
        )

        plt.tight_layout()

        _save(
            "correlation_heatmap.png"
        )

        charts.append(
            "correlation_heatmap.png"
        )

    # =========================================================
    # 16. MISSING VALUES DICTIONARY
    # =========================================================

    missing_dict = {
        str(col): int(cnt)
        for col, cnt in missing.items()
        if cnt > 0
    }


    # =========================================================
    # 17. NUMERIC SUMMARY
    # =========================================================

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


    # =========================================================
    # 18. DATASET SOURCE COUNTS
    # =========================================================

    if "_dataset_source" in data.columns:

        source_counts = (
            data["_dataset_source"]
            .value_counts()
            .to_dict()
        )

    else:

        source_counts = {}


    # =========================================================
    # 19. RETURN RESULTS
    # =========================================================

    return {
        "n_rows": len(data),
        "n_cols": len(original_columns),

        "duplicate_count": duplicate_count,
        "duplicate_id_count": duplicate_id_count,

        "missing": missing_dict,

        "outlier_counts": outlier_counts,
        "outlier_limits": outlier_limits,

        "source_counts": {
            str(k): int(v)
            for k, v in source_counts.items()
        },

        "numeric_summary": numeric_summary,

        "charts": charts
    }

    if __name__ == "__main__":

        result = run_eda()

        print("\n========== EDA SUMMARY ==========")

        print(
            "Rows:",
            result["n_rows"]
        )

        print(
            "Columns:",
            result["n_cols"]
        )

        print(
            "Duplicate Rows:",
            result["duplicate_count"]
        )

        print(
            "Duplicate IDs:",
            result["duplicate_id_count"]
        )

        print("\nOutliers:")

        for col, count in result[
            "outlier_counts"
        ].items():
            print(
                f"{col}: {count:,}"
            )

        print("\nCharts:")

        for chart in result["charts"]:
            print(chart)