import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from load_data import load_data
from preprocessing import extract_engineered_features

CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "charts")

# Visual Styling constants matching the UI reference
SERIES_COLOR = "#5ba0d7"       # Soft Sky-Blue series color
TITLE_COLOR = "#1f3a5f"        # Deep navy blue title color
LABEL_COLOR = "#4a5568"        # Crisp slate label color
GRID_COLOR = "#e2e8f0"         # Subtle horizontal gridlines


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _apply_theme(ax, title: str, xlabel: str, ylabel: str):
    """Apply unified, publication-grade styling matching the dashboard reference."""
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=TITLE_COLOR, pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=LABEL_COLOR, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, color=LABEL_COLOR, labelpad=8)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cbd5e1")
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors=LABEL_COLOR, labelsize=9.5)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")


EDA_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "eda_metrics.pkl")


def run_eda(force_regenerate: bool = False) -> dict:
    """
    Generate all 18 vertical EDA visualizations with unified sky-blue theme
    and calculate rigorous statistical distributions and outlier limits.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EDA_CACHE_PATH), exist_ok=True)

    # Check fast disk cache
    if not force_regenerate and os.path.exists(EDA_CACHE_PATH):
        try:
            with open(EDA_CACHE_PATH, "rb") as f:
                cached_eda = pickle.load(f)
            # Verify missing_values.png and price_distribution.png exist on disk
            if os.path.exists(_chart_path("missing_values.png")) and os.path.exists(_chart_path("price_distribution.png")):
                return cached_eda
        except Exception:
            pass

    data = load_data()
    # Extract engineered features for rich amenity analysis
    data_eng = extract_engineered_features(data.copy())

    charts = []

    # =========================================================
    # 1. MISSING VALUES BY COLUMN
    # =========================================================
    missing_pct = (data.isnull().sum() / len(data)) * 100
    missing_df = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if not missing_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        bars = ax.bar(range(len(missing_df)), missing_df.values, color=SERIES_COLOR, width=0.6, edgecolor="#3b82f6", linewidth=0.5)
        ax.set_xticks(range(len(missing_df)))
        ax.set_xticklabels(missing_df.index, rotation=35, ha="right")
        _apply_theme(ax, "Missing Values by Column", "Feature / Attribute", "Percentage of Missing Values (%)")
        _save("missing_values.png")
        charts.append("missing_values.png")

    # Clean numeric series for statistics
    price = pd.to_numeric(data["price"], errors="coerce").dropna()
    valid_price = price[(price > 100) & (price <= 6000)]
    sqft = pd.to_numeric(data["square_feet"], errors="coerce").dropna()
    valid_sqft = sqft[(sqft > 100) & (sqft <= 4000)]

    # =========================================================
    # 2. RENTAL PRICE DISTRIBUTION
    # =========================================================
    if not valid_price.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.hist(valid_price, bins=45, color=SERIES_COLOR, edgecolor="white", alpha=0.9)
        _apply_theme(ax, "Rental Price Distribution ($100 - $6,000 Range)", "Monthly Rent ($ USD)", "Frequency (Listings)")
        _save("price_distribution.png")
        charts.append("price_distribution.png")

    # =========================================================
    # 3. RENTAL PRICE BOXPLOT
    # =========================================================
    if not valid_price.empty:
        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="white")
        bp = ax.boxplot(valid_price, vert=False, patch_artist=True,
                        boxprops=dict(facecolor=SERIES_COLOR, color="#1f3a5f"),
                        medianprops=dict(color="#b91c1c", linewidth=2),
                        whiskerprops=dict(color="#1f3a5f"),
                        capprops=dict(color="#1f3a5f"),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="#93c5fd", alpha=0.5))
        _apply_theme(ax, "Rental Price Distribution (Boxplot & Outliers)", "Monthly Rent ($ USD)", "")
        ax.set_yticks([])
        _save("price_boxplot.png")
        charts.append("price_boxplot.png")

    # =========================================================
    # 4. SQUARE FEET DISTRIBUTION
    # =========================================================
    if not valid_sqft.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.hist(valid_sqft, bins=40, color=SERIES_COLOR, edgecolor="white", alpha=0.9)
        _apply_theme(ax, "Living Area Distribution (Square Feet)", "Square Feet", "Listing Count")
        _save("square_feet_distribution.png")
        charts.append("square_feet_distribution.png")

    # =========================================================
    # 5. PRICE VS SQUARE FEET
    # =========================================================
    comb = data[["price", "square_feet"]].dropna()
    comb["price"] = pd.to_numeric(comb["price"], errors="coerce")
    comb["square_feet"] = pd.to_numeric(comb["square_feet"], errors="coerce")
    comb = comb.dropna()
    comb_filt = comb[(comb["price"] > 200) & (comb["price"] <= 6000) & (comb["square_feet"] > 200) & (comb["square_feet"] <= 3500)]
    if not comb_filt.empty:
        sample_comb = comb_filt.sample(min(len(comb_filt), 3500), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.scatter(sample_comb["square_feet"], sample_comb["price"], alpha=0.35, color=SERIES_COLOR, edgecolors="none", s=22)
        # Add linear trendline
        m, b = np.polyfit(sample_comb["square_feet"], sample_comb["price"], 1)
        xs = np.linspace(sample_comb["square_feet"].min(), sample_comb["square_feet"].max(), 100)
        ax.plot(xs, m * xs + b, color="#1f3a5f", linewidth=2, label="Linear Trend")
        ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
        _apply_theme(ax, "Rental Price vs Living Area (Square Footage)", "Square Feet", "Monthly Rent ($ USD)")
        _save("price_vs_square_feet.png")
        charts.append("price_vs_square_feet.png")

    # =========================================================
    # 6. BEDROOMS DISTRIBUTION
    # =========================================================
    if "bedrooms" in data.columns:
        bed_counts = data["bedrooms"].value_counts().sort_index()
        bed_counts = bed_counts[(bed_counts.index >= 0) & (bed_counts.index <= 6)]
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar(bed_counts.index.astype(str), bed_counts.values, color=SERIES_COLOR, width=0.55, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Distribution of Listings by Bedroom Count", "Number of Bedrooms", "Number of Listings")
        _save("bedrooms_distribution.png")
        charts.append("bedrooms_distribution.png")

    # =========================================================
    # 7. BATHROOMS DISTRIBUTION
    # =========================================================
    if "bathrooms" in data.columns:
        bath_counts = data["bathrooms"].value_counts().sort_index()
        bath_counts = bath_counts[(bath_counts.index >= 1) & (bath_counts.index <= 5)]
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar(bath_counts.index.astype(str), bath_counts.values, color=SERIES_COLOR, width=0.55, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Distribution of Listings by Bathroom Count", "Number of Bathrooms", "Number of Listings")
        _save("bathrooms_distribution.png")
        charts.append("bathrooms_distribution.png")

    # =========================================================
    # 8. PRICE BY BEDROOMS (MEDIAN)
    # =========================================================
    if "bedrooms" in data.columns and "price" in data.columns:
        bed_price = data[["bedrooms", "price"]].dropna()
        bed_price["bedrooms"] = pd.to_numeric(bed_price["bedrooms"], errors="coerce")
        bed_price["price"] = pd.to_numeric(bed_price["price"], errors="coerce")
        bed_price = bed_price.dropna()
        bed_med = bed_price[(bed_price["bedrooms"] >= 0) & (bed_price["bedrooms"] <= 5) & (bed_price["price"] > 100) & (bed_price["price"] <= 10000)].groupby("bedrooms")["price"].median()
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar([f"{int(b)} Bed" for b in bed_med.index], bed_med.values, color=SERIES_COLOR, width=0.55, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Median Monthly Rent by Bedroom Count", "Bedrooms", "Median Rent ($ USD)")
        for i, val in enumerate(bed_med.values):
            ax.text(i, val + 25, f"${int(val):,}", ha="center", fontsize=9, fontweight="bold", color=TITLE_COLOR)
        _save("price_by_bedrooms.png")
        charts.append("price_by_bedrooms.png")

    # =========================================================
    # 9. PRICE BY BATHROOMS (MEDIAN)
    # =========================================================
    if "bathrooms" in data.columns and "price" in data.columns:
        bath_price = data[["bathrooms", "price"]].dropna()
        bath_price["bathrooms"] = pd.to_numeric(bath_price["bathrooms"], errors="coerce")
        bath_price["price"] = pd.to_numeric(bath_price["price"], errors="coerce")
        bath_price = bath_price.dropna()
        bath_med = bath_price[(bath_price["bathrooms"] >= 1) & (bath_price["bathrooms"] <= 4) & (bath_price["price"] > 100) & (bath_price["price"] <= 10000)].groupby("bathrooms")["price"].median()
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar([f"{b:g} Bath" for b in bath_med.index], bath_med.values, color=SERIES_COLOR, width=0.55, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Median Monthly Rent by Bathroom Count", "Bathrooms", "Median Rent ($ USD)")
        for i, val in enumerate(bath_med.values):
            ax.text(i, val + 25, f"${int(val):,}", ha="center", fontsize=9, fontweight="bold", color=TITLE_COLOR)
        _save("price_by_bathrooms.png")
        charts.append("price_by_bathrooms.png")

    # =========================================================
    # 10. TOP STATES BY LISTING VOLUME
    # =========================================================
    if "state" in data.columns:
        top_states = data["state"].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar(top_states.index, top_states.values, color=SERIES_COLOR, width=0.6, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Top 12 US States by Listing Volume", "State Code", "Number of Listings")
        _save("top_states.png")
        charts.append("top_states.png")

    # =========================================================
    # 11. PRICE BY STATE (MEDIAN IN TOP STATES)
    # =========================================================
    if "state" in data.columns and "price" in data.columns:
        state_df = data[["state", "price"]].dropna()
        state_df["price"] = pd.to_numeric(state_df["price"], errors="coerce")
        state_df = state_df[(state_df["price"] > 100) & (state_df["price"] <= 10000)]
        top_states_list = data["state"].value_counts().head(12).index
        state_med = state_df[state_df["state"].isin(top_states_list)].groupby("state")["price"].median().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar(state_med.index, state_med.values, color=SERIES_COLOR, width=0.6, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Median Monthly Rent across Top 12 States", "State Code", "Median Rent ($ USD)")
        for i, val in enumerate(state_med.values):
            ax.text(i, val + 20, f"${int(val):,}", ha="center", fontsize=8.5, fontweight="bold", color=TITLE_COLOR)
        _save("price_by_state.png")
        charts.append("price_by_state.png")

    # =========================================================
    # 12. PET POLICY DISTRIBUTION
    # =========================================================
    if "pets_allowed" in data.columns:
        pet_s = data["pets_allowed"].fillna("None").astype(str).str.strip()
        pet_mapped = pet_s.replace({
            "Cats,Dogs": "Cats & Dogs Allowed",
            "Cats": "Cats Only",
            "Dogs": "Dogs Only",
            "None": "No Pets / Not Specified",
            "": "No Pets / Not Specified"
        })
        pet_counts = pet_mapped.value_counts().head(4)
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar(pet_counts.index, pet_counts.values, color=SERIES_COLOR, width=0.55, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Distribution of Pet Policies", "Pet Policy", "Listing Count")
        _save("pets_allowed.png")
        charts.append("pets_allowed.png")

    # =========================================================
    # 13. PRICE BY PET POLICY
    # =========================================================
    if "pets_allowed" in data.columns and "price" in data.columns:
        pet_price_df = data[["pets_allowed", "price"]].copy()
        pet_price_df["price"] = pd.to_numeric(pet_price_df["price"], errors="coerce")
        pet_price_df["pet_cat"] = pet_price_df["pets_allowed"].fillna("None").astype(str).str.strip().replace({
            "Cats,Dogs": "Cats & Dogs",
            "Cats": "Cats Only",
            "Dogs": "Dogs Only",
            "None": "None / Unspecified"
        })
        pet_price_df = pet_price_df[(pet_price_df["price"] > 100) & (pet_price_df["price"] <= 8000)]
        pet_med = pet_price_df.groupby("pet_cat")["price"].median().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar(pet_med.index, pet_med.values, color=SERIES_COLOR, width=0.55, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Median Monthly Rent by Pet Policy", "Pet Policy", "Median Rent ($ USD)")
        for i, val in enumerate(pet_med.values):
            ax.text(i, val + 20, f"${int(val):,}", ha="center", fontsize=9, fontweight="bold", color=TITLE_COLOR)
        _save("price_by_pets.png")
        charts.append("price_by_pets.png")

    # =========================================================
    # 14. LISTINGS WITH PHOTO VS WITHOUT
    # =========================================================
    if "has_photo" in data.columns:
        photo_counts = data["has_photo"].fillna("No").value_counts()
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar(photo_counts.index.astype(str), photo_counts.values, color=SERIES_COLOR, width=0.45, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Listings With Verified Photos vs Without", "Has Photo", "Listing Count")
        _save("has_photo.png")
        charts.append("has_photo.png")

    # =========================================================
    # 15. PRICE BY PHOTO STATUS
    # =========================================================
    if "has_photo" in data.columns and "price" in data.columns:
        hp_df = data[["has_photo", "price"]].dropna()
        hp_df["price"] = pd.to_numeric(hp_df["price"], errors="coerce")
        hp_df = hp_df[(hp_df["price"] > 100) & (hp_df["price"] <= 8000)]
        hp_med = hp_df.groupby("has_photo")["price"].median()
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar(hp_med.index.astype(str), hp_med.values, color=SERIES_COLOR, width=0.45, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Median Rental Price by Photo Status", "Has Photo", "Median Rent ($ USD)")
        for i, val in enumerate(hp_med.values):
            ax.text(i, val + 20, f"${int(val):,}", ha="center", fontsize=9, fontweight="bold", color=TITLE_COLOR)
        _save("price_by_photo.png")
        charts.append("price_by_photo.png")

    # =========================================================
    # 16. CORRELATION HEATMAP
    # =========================================================
    num_cols = ["price", "square_feet", "bedrooms", "bathrooms", "latitude", "longitude"]
    num_df = data[[c for c in num_cols if c in data.columns]].apply(pd.to_numeric, errors="coerce").dropna()
    if not num_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
        corr = num_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax, linewidths=0.5, linecolor="#e2e8f0")
        ax.set_title("Correlation Heatmap (Continuous Features)", fontsize=13, fontweight="bold", color=TITLE_COLOR, pad=12)
        _save("correlation_heatmap.png")
        charts.append("correlation_heatmap.png")

    # =========================================================
    # 17. NEW: RENTAL PRICE BY KEY AMENITIES (ENGINEERED)
    # =========================================================
    amenity_cols = [
        ("has_pool", "Pool"),
        ("has_gym", "Fitness Gym"),
        ("has_washer_dryer", "In-Unit Laundry"),
        ("has_ac", "Central A/C"),
        ("has_elevator", "Elevator"),
        ("has_dishwasher", "Dishwasher"),
        ("has_parking", "Dedicated Parking"),
        ("feat_luxury", "Luxury Finishes")
    ]
    amenity_premiums = []
    for col, label in amenity_cols:
        if col in data_eng.columns and "price" in data_eng.columns:
            sub = data_eng[[col, "price"]].dropna()
            sub["price"] = pd.to_numeric(sub["price"], errors="coerce")
            sub = sub[(sub["price"] > 100) & (sub["price"] <= 8000)]
            med_with = sub[sub[col] == 1]["price"].median()
            med_without = sub[sub[col] == 0]["price"].median()
            diff = med_with - med_without
            amenity_premiums.append({"Amenity": label, "With": med_with, "Without": med_without, "Premium": diff})

    if amenity_premiums:
        prem_df = pd.DataFrame(amenity_premiums).sort_values("Premium", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
        ax.bar(prem_df["Amenity"], prem_df["Premium"], color=SERIES_COLOR, width=0.55, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Median Rent Premium for Listings Offering Key Amenities", "Engineered Amenity Feature", "Monthly Rent Premium (+$ USD)")
        ax.set_xticks(range(len(prem_df)))
        ax.set_xticklabels(prem_df["Amenity"], rotation=30, ha="right")
        for i, val in enumerate(prem_df["Premium"].values):
            ax.text(i, val + 5, f"+${int(val)}", ha="center", fontsize=9, fontweight="bold", color=TITLE_COLOR)
        _save("price_by_amenities.png")
        charts.append("price_by_amenities.png")

    # =========================================================
    # 18. NEW: TOTAL AMENITIES COUNT DISTRIBUTION
    # =========================================================
    if "amenity_count" in data_eng.columns:
        cnt_dist = data_eng["amenity_count"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
        ax.bar(cnt_dist.index.astype(str), cnt_dist.values, color=SERIES_COLOR, width=0.6, edgecolor="#3b82f6", linewidth=0.5)
        _apply_theme(ax, "Distribution of Total Amenities Count per Listing", "Number of Tracked Amenities (0 - 13)", "Listing Frequency")
        _save("amenities_distribution.png")
        charts.append("amenities_distribution.png")

    # Compute Statistical Metrics & Outlier Limits
    outlier_counts = {}
    outlier_limits = {}
    for col in ["price", "square_feet", "bedrooms", "bathrooms"]:
        if col in data.columns:
            vals = pd.to_numeric(data[col], errors="coerce").dropna()
            if not vals.empty:
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = vals[(vals < lower) | (vals > upper)]
                outlier_counts[col] = int(len(outliers))
                outlier_limits[col] = {
                    "q1": round(float(q1), 2),
                    "q3": round(float(q3), 2),
                    "iqr": round(float(iqr), 2),
                    "lower": round(float(lower), 2),
                    "upper": round(float(upper), 2)
                }

    result_dict = {
        "charts": charts,
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "outlier_counts": outlier_counts,
        "outlier_limits": outlier_limits,
        "price_stats": {
            "mean": round(float(valid_price.mean()), 2) if not valid_price.empty else 0,
            "median": round(float(valid_price.median()), 2) if not valid_price.empty else 0,
            "std": round(float(valid_price.std()), 2) if not valid_price.empty else 0,
            "min": round(float(valid_price.min()), 2) if not valid_price.empty else 0,
            "max": round(float(valid_price.max()), 2) if not valid_price.empty else 0
        },
        "sqft_stats": {
            "mean": round(float(valid_sqft.mean()), 2) if not valid_sqft.empty else 0,
            "median": round(float(valid_sqft.median()), 2) if not valid_sqft.empty else 0,
            "std": round(float(valid_sqft.std()), 2) if not valid_sqft.empty else 0,
            "min": round(float(valid_sqft.min()), 2) if not valid_sqft.empty else 0,
            "max": round(float(valid_sqft.max()), 2) if not valid_sqft.empty else 0
        }
    }

    try:
        with open(EDA_CACHE_PATH, "wb") as f:
            pickle.dump(result_dict, f)
    except Exception as e:
        print(f"Notice saving EDA cache: {e}")

    return result_dict


if __name__ == "__main__":
    print("Testing apartment_eda.py...")
    res = run_eda()
    print("Generated charts:", len(res["charts"]))
    print(res["charts"])