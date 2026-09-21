import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from knee_for_DBSCAN import find_eps
from load_data import load_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

CLUSTER_FEATURES = [
    "price",
    "bedrooms",
    "bathrooms",
    "square_feet",
    "latitude",
    "longitude"
]


def prepare_dbscan_data(sample_size=1000):
    """
    Load RentRadar dataset, clean numerical features, apply sample size cap (max 1000),
    and standardize features using StandardScaler for DBSCAN distance calculation.
    """
    data = load_data().copy()

    if "_dataset_source" in data.columns:
        data = data.drop(columns=["_dataset_source"])

    available_features = [col for col in CLUSTER_FEATURES if col in data.columns]
    if len(available_features) < 2:
        raise ValueError("Not enough numerical features available for DBSCAN clustering.")

    X = data[available_features].copy()

    for col in available_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    if len(X) < 10:
        raise ValueError("Not enough valid records available for DBSCAN clustering.")

    total_records = len(X)

    # Apply maximum 1000 samples cap
    if total_records > sample_size:
        data_sampled = data.loc[X.index].sample(n=sample_size, random_state=42)
        X_sampled = X.loc[data_sampled.index].copy()
    else:
        data_sampled = data.loc[X.index].copy()
        X_sampled = X.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sampled)

    return data_sampled, X_sampled, X_scaled, available_features, total_records


def plot_dbscan_clusters(X_df, X_scaled, labels, core_indices, filename="dbscan_clusters.png"):
    """
    Generate and save a 2D scatter plot visualizing DBSCAN clusters,
    distinguishing Core Points, Border Points, and Noise Points.
    """
    plt.figure(figsize=(10, 6.5))

    # Determine 2 features for plotting (e.g. Price vs Square Feet or Longitude vs Latitude)
    if "square_feet" in X_df.columns and "price" in X_df.columns:
        x_col, y_col = "square_feet", "price"
    else:
        x_col, y_col = X_df.columns[0], X_df.columns[1]

    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    core_mask = np.zeros_like(labels, dtype=bool)
    core_mask[core_indices] = True
    noise_mask = (labels == -1)

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0.6, 0.6, 0.6, 0.6]  # Grey for noise
            class_member_mask = noise_mask
            plt.scatter(
                X_df.loc[class_member_mask, x_col],
                X_df.loc[class_member_mask, y_col],
                c=[col],
                marker="x",
                s=40,
                alpha=0.7,
                label="Noise (-1)"
            )
        else:
            class_member_mask = (labels == k)
            # Plot Core Points
            plt.scatter(
                X_df.loc[class_member_mask & core_mask, x_col],
                X_df.loc[class_member_mask & core_mask, y_col],
                c=[col],
                marker="o",
                s=60,
                edgecolor="k",
                linewidth=0.5,
                alpha=0.9,
                label=f"Cluster {k+1} (Core)"
            )
            # Plot Border Points
            plt.scatter(
                X_df.loc[class_member_mask & ~core_mask, x_col],
                X_df.loc[class_member_mask & ~core_mask, y_col],
                c=[col],
                marker="s",
                s=35,
                alpha=0.7,
                edgecolor="w",
                linewidth=0.5
            )

    plt.xlabel(x_col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    plt.ylabel(y_col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    plt.title("DBSCAN Clustering (Core, Border & Noise Distribution)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, frameon=True)
    plt.tight_layout()

    chart_path = os.path.join(CHART_DIR, filename)
    plt.savefig(chart_path, dpi=150)
    plt.close()

    return f"charts/{filename}"


def run_dbscan(eps="auto", min_samples=5):
    """
    DBSCAN Execution Pipeline for RentRadar:
    1. Loads dataset, caps at 1000 samples, standardizes features.
    2. Uses knee_for_DBSCAN.find_eps to dynamically derive optimal eps if eps='auto'.
    3. Fits DBSCAN(eps=eps, min_samples=min_samples).
    4. Identifies Core, Border, and Noise points.
    5. Calculates cluster statistics, Silhouette Score, and generates visualization charts.
    """
    min_samples = int(min_samples) if min_samples else 5
    if min_samples < 2:
        min_samples = 2

    raw_data, X_df, X_scaled, features, total_records = prepare_dbscan_data(sample_size=1000)

    # 1. Determine Epsilon (eps)
    is_auto = False
    if eps == "auto" or eps is None or str(eps).lower() == "auto":
        derived_eps = find_eps(X_scaled, min_samples=min_samples, save_plot=True, filename="dbscan_k_distance.png")
        selected_eps = derived_eps
        is_auto = True
    else:
        try:
            selected_eps = float(eps)
            if selected_eps <= 0:
                selected_eps = 0.5
        except (ValueError, TypeError):
            selected_eps = find_eps(X_scaled, min_samples=min_samples, save_plot=True, filename="dbscan_k_distance.png")
            is_auto = True
        derived_eps = find_eps(X_scaled, min_samples=min_samples, save_plot=False)

    # Ensure K-Distance Graph exists even if custom eps is passed
    k_distance_chart = "charts/dbscan_k_distance.png"
    if not os.path.exists(os.path.join(CHART_DIR, "dbscan_k_distance.png")):
        find_eps(X_scaled, min_samples=min_samples, save_plot=True, filename="dbscan_k_distance.png")

    # 2. Build and Fit DBSCAN model
    model = DBSCAN(eps=selected_eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    # 3. Core, Border, and Noise Points identification
    core = model.core_sample_indices_
    noise = (labels == -1)
    border = ~noise & ~pd.Series(range(len(X_scaled))).isin(core)

    unique_labels = set(labels)
    clusters_count = len(unique_labels) - (1 if -1 in labels else 0)
    core_count = len(core)
    border_count = int(border.sum())
    noise_count = int(noise.sum())

    # Print results (matching user's reference print statements)
    print("Clusters:", clusters_count)
    print("Core:", core_count)
    print("Border:", border_count)
    print("Noise:", noise_count)

    # 4. Compute Metrics & Summaries
    sil_score = None
    non_noise_mask = ~noise
    if clusters_count >= 2 and non_noise_mask.sum() > clusters_count:
        try:
            sil_score = round(float(silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])), 4)
        except Exception:
            sil_score = None

    X_df_result = X_df.copy()
    X_df_result["cluster"] = labels

    cluster_summaries = []

    for k in sorted(list(unique_labels)):
        c_mask = (labels == k)
        c_data = X_df_result[c_mask]
        c_core = c_mask & pd.Series(range(len(X_scaled))).isin(core)
        c_border = c_mask & border

        cluster_name = f"Cluster {k+1}" if k != -1 else "Noise Points (-1)"
        summary_item = {
            "cluster_id": k if k == -1 else k + 1,
            "name": cluster_name,
            "is_noise": (k == -1),
            "count": int(c_mask.sum()),
            "core_count": int(c_core.sum()),
            "border_count": int(c_border.sum()),
            "avg_price": round(c_data["price"].mean(), 2) if "price" in c_data and len(c_data) > 0 else 0,
            "avg_sqft": round(c_data["square_feet"].mean(), 2) if "square_feet" in c_data and len(c_data) > 0 else 0,
            "avg_beds": round(c_data["bedrooms"].mean(), 1) if "bedrooms" in c_data and len(c_data) > 0 else 0,
            "avg_baths": round(c_data["bathrooms"].mean(), 1) if "bathrooms" in c_data and len(c_data) > 0 else 0,
        }
        cluster_summaries.append(summary_item)

    # 5. Generate Cluster Plot
    cluster_chart = plot_dbscan_clusters(X_df, X_scaled, labels, core)

    results = {
        "eps": selected_eps,
        "derived_eps": derived_eps,
        "is_auto": is_auto,
        "min_samples": min_samples,
        "clusters_count": clusters_count,
        "core_count": core_count,
        "border_count": border_count,
        "noise_count": noise_count,
        "silhouette_score": sil_score,
        "total_records": total_records,
        "sampled_records": len(X_scaled),
        "cluster_summaries": cluster_summaries,
        "k_distance_chart": k_distance_chart,
        "cluster_chart": cluster_chart,
        "features": features
    }

    return results


if __name__ == "__main__":
    print("\n--- Executing DBSCAN.py Standalone ---")

    # If reference placement dataset exists, test on placement dataset first
    placement_csv = os.path.join(BASE_DIR, "preprocessed_placement.csv")
    if os.path.exists(placement_csv):
        data = pd.read_csv(placement_csv)
        if len(data) > 1000:
            data = data.sample(n=1000, random_state=42)
        X = data.drop("PlacementStatus", axis=1) if "PlacementStatus" in data.columns else data.copy()

        model = DBSCAN(eps=find_eps(X, min_samples=5), min_samples=5)
        labels = model.fit_predict(X)
        core = model.core_sample_indices_
        noise = labels == -1
        border = ~noise & ~pd.Series(range(len(X))).isin(core)

        print("Clusters:", len(set(labels)) - (1 if -1 in labels else 0))
        print("Core:", len(core))
        print("Border:", border.sum())
        print("Noise:", noise.sum())

        plt.scatter(
            X.iloc[:, 0],
            X.iloc[:, 1],
            c=labels
        )
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.title("DBSCAN Clustering")
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, "dbscan_standalone_placement.png"))
        plt.close()
    else:
        # Run RentRadar pipeline
        res = run_dbscan(eps="auto", min_samples=5)
        print("RentRadar DBSCAN pipeline output success.")
