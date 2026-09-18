import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from load_data import load_data


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")

os.makedirs(CHART_DIR, exist_ok=True)


# =========================================================
# CONFIGURATION
# =========================================================

CLUSTER_FEATURES = [
    "price",
    "bedrooms",
    "bathrooms",
    "square_feet",
    "latitude",
    "longitude"
]

MAX_SAMPLE_SIZE = 2500
RANDOM_STATE = 42


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_hierarchical_data(sample_size=MAX_SAMPLE_SIZE):
    """
    Load dataset, clean numerical features, scale data,
    and return a sampled subset suitable for hierarchical clustering.
    """
    print("Loading dataset for Hierarchical Clustering...")
    data = load_data().copy()

    if "_dataset_source" in data.columns:
        data = data.drop(columns=["_dataset_source"])

    available_features = [col for col in CLUSTER_FEATURES if col in data.columns]
    if len(available_features) < 2:
        raise ValueError("Not enough numerical features available for Hierarchical Clustering.")

    X = data[available_features].copy()

    for col in available_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    if len(X) < 10:
        raise ValueError("Not enough valid records available for Hierarchical Clustering.")

    total_valid = len(X)

    # Sample for performance if dataset is large
    if total_valid > sample_size:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_indices = rng.choice(total_valid, size=sample_size, replace=False)
        X_sample = X.iloc[sample_indices].copy()
        raw_sample = data.loc[X_sample.index].copy()
    else:
        X_sample = X.copy()
        raw_sample = data.loc[X_sample.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    return raw_sample, X_sample, X_scaled, available_features, total_valid


# =========================================================
# PLOTTING HELPERS
# =========================================================

def plot_dendrogram(Z, n_clusters, filename="hierarchical_dendrogram.png"):
    """
    Generate and save a truncated dendrogram plot with explicit height annotations at merge levels.
    """
    plt.figure(figsize=(12, 6.5))
    plt.title("Hierarchical Clustering Dendrogram with Level Merge Heights", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Sample Index / Cluster Size", fontsize=11)
    plt.ylabel("Distance (Dissimilarity / Merge Height)", fontsize=11)

    cut_height = float(Z[-(n_clusters - 1), 2]) if n_clusters > 1 and len(Z) >= (n_clusters - 1) else 0.0

    ddata = dendrogram(
        Z,
        truncate_mode="level",
        p=5,
        leaf_rotation=45.0,
        leaf_font_size=10.0,
        show_contracted=True,
        color_threshold=cut_height if n_clusters > 1 else None
    )

    # Annotate height values on the dendrogram branches
    max_y = max([max(y) for y in ddata['dcoord']]) if ddata['dcoord'] else 10.0
    min_label_y = max_y * 0.12  # Only label merge bars above 12% of max height to keep plot clean

    annotated_coords = set()

    for i_coord, d_coord, color in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
        x = 0.5 * (i_coord[1] + i_coord[2])
        y = d_coord[1]

        coord_key = (round(x, 1), round(y, 1))
        if y >= min_label_y and coord_key not in annotated_coords:
            annotated_coords.add(coord_key)
            plt.annotate(
                f"h={y:.1f}",
                (x, y),
                xytext=(0, 4),
                textcoords="offset points",
                va="bottom",
                ha="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor=color, alpha=0.85, linewidth=0.8)
            )

    # Draw cut height line
    plt.axhline(
        y=cut_height,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Cut Height = {cut_height:.2f} (K={n_clusters})"
    )

    # Label cut line on chart right edge
    xlim = plt.xlim()
    plt.text(
        xlim[1] * 0.98,
        cut_height,
        f" Cut Height: {cut_height:.2f}",
        color="red",
        fontweight="bold",
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="square,pad=0.2", facecolor="#fff0f0", edgecolor="red", alpha=0.9)
    )

    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    chart_path = os.path.join(CHART_DIR, filename)
    plt.savefig(chart_path, dpi=150)
    plt.close()

    return f"charts/{filename}"


def plot_cluster_scatter(X_df, labels, n_clusters, filename="hierarchical_clusters.png"):
    """
    Generate and save 2D scatter plot of Price vs Square Feet by Cluster.
    """
    plt.figure(figsize=(10, 6))

    try:
        colors = plt.colormaps["tab10"]
    except Exception:
        colors = plt.cm.tab10

    for c in range(n_clusters):
        cluster_mask = (labels == c)
        plt.scatter(
            X_df.loc[cluster_mask, "square_feet"],
            X_df.loc[cluster_mask, "price"],
            s=35,
            alpha=0.7,
            color=colors(c % 10),
            label=f"Cluster {c + 1}"
        )

    plt.title("Hierarchical Clusters: Rental Price vs Square Feet", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Square Feet", fontsize=11)
    plt.ylabel("Monthly Rent ($)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(title="Clusters", loc="upper left")
    plt.tight_layout()

    chart_path = os.path.join(CHART_DIR, filename)
    plt.savefig(chart_path, dpi=150)
    plt.close()

    return f"charts/{filename}"


# =========================================================
# MAIN EXECUTION PIPELINE
# =========================================================

def run_hierarchical_clustering(n_clusters="auto", linkage_method="ward", metric="euclidean"):
    """
    Hierarchical Clustering Pipeline:
    1. Loads dataset & scales features using StandardScaler.
    2. Builds the complete Agglomerative Dendrogram / Linkage Hierarchy Matrix Z FIRST using Euclidean metric.
    3. Analyzes the dendrogram's merge heights (distance gaps) to automatically derive the optimal K.
    4. Applies the cut threshold (either automatically derived K or manual selection).
    5. Calculates Cophenetic Correlation, Silhouette Score, and generates Dendrogram & Cluster Scatter plots.
    """
    metric = "euclidean"

    # Sanitize linkage_method
    if not linkage_method or not isinstance(linkage_method, str):
        linkage_method = "ward"
    linkage_method = linkage_method.lower().strip()
    valid_linkages = ["ward", "complete", "average", "single"]
    if linkage_method not in valid_linkages:
        linkage_method = "ward"

    raw_data, X_df, X_scaled, features, total_valid = prepare_hierarchical_data()

    # 1. Build Complete Hierarchy Tree FIRST (Bottom-Up Agglomerative Linkage)
    print(f"Building complete hierarchy tree using {linkage_method} linkage and Euclidean distance...")
    Z = linkage(X_scaled, method=linkage_method, metric="euclidean")

    # 2. Cophenetic Correlation Coefficient
    print("Calculating Cophenetic Correlation Coefficient...")
    coph_matrix = pdist(X_scaled, metric="euclidean")
    coph_coeff, _ = cophenet(Z, coph_matrix)

    # 3. Analyze Dendrogram Merge Heights & Derive Optimal K
    n_samples = len(X_scaled)
    max_k_eval = min(10, n_samples)
    level_heights = []

    for k in range(2, max_k_eval + 1):
        idx = -(k - 1)
        if abs(idx) <= len(Z):
            h = float(Z[idx, 2])
            level_heights.append({
                "k": k,
                "height": round(h, 2),
                "is_current": False
            })

    # Derive optimal K from maximum vertical distance gap (largest height jump) in dendrogram
    best_k = 3
    max_gap = 0.0

    for i in range(len(level_heights) - 1):
        k_curr = level_heights[i]["k"]
        h_curr = level_heights[i]["height"]
        h_next = level_heights[i + 1]["height"]
        gap = h_curr - h_next

        if gap > max_gap:
            max_gap = gap
            best_k = k_curr

    derived_k = best_k

    # Handle requested n_clusters (auto vs int)
    is_auto = False
    if n_clusters == "auto" or n_clusters is None or str(n_clusters).lower() == "auto":
        selected_k = derived_k
        is_auto = True
    else:
        try:
            selected_k = int(n_clusters)
            if selected_k < 2:
                selected_k = 2
            elif selected_k > 10:
                selected_k = 10
        except (TypeError, ValueError):
            selected_k = derived_k
            is_auto = True

    # Mark current selected K in level_heights
    for item in level_heights:
        if item["k"] == selected_k:
            item["is_current"] = True

    # 4. Apply Cut Line to Dendrogram Hierarchy for selected_k
    print(f"Applying cut line to dendrogram for K={selected_k} (Derived Optimal K={derived_k})...")
    model = AgglomerativeClustering(
        n_clusters=selected_k,
        metric="euclidean",
        linkage=linkage_method
    )
    labels = model.fit_predict(X_scaled)

    # 5. Silhouette Score & Cluster Summaries
    sil_score = silhouette_score(X_scaled, labels)

    X_df_result = X_df.copy()
    X_df_result["cluster"] = labels + 1

    cluster_sizes = {}
    cluster_summaries = []

    for c in range(1, selected_k + 1):
        c_mask = (X_df_result["cluster"] == c)
        count = int(c_mask.sum())
        cluster_sizes[c] = count

        c_data = X_df_result[c_mask]
        summary_item = {
            "cluster": c,
            "count": count,
            "avg_price": round(c_data["price"].mean(), 2) if "price" in c_data else 0,
            "avg_sqft": round(c_data["square_feet"].mean(), 2) if "square_feet" in c_data else 0,
            "avg_beds": round(c_data["bedrooms"].mean(), 1) if "bedrooms" in c_data else 0,
            "avg_baths": round(c_data["bathrooms"].mean(), 1) if "bathrooms" in c_data else 0,
        }
        cluster_summaries.append(summary_item)

    cut_height = round(float(Z[-(selected_k - 1), 2]), 2) if selected_k > 1 and len(Z) >= (selected_k - 1) else 0.0

    # 6. Generate Charts
    dendrogram_chart = plot_dendrogram(Z, selected_k)
    cluster_chart = plot_cluster_scatter(X_df_result, labels, selected_k)

    results = {
        "n_clusters": selected_k,
        "derived_k": derived_k,
        "is_auto": is_auto,
        "max_gap": round(float(max_gap), 2),
        "linkage_method": linkage_method,
        "metric": "euclidean",
        "silhouette_score": round(float(sil_score), 4),
        "cophenetic_coeff": round(float(coph_coeff), 4),
        "cut_height": cut_height,
        "level_heights": level_heights,
        "total_records": total_valid,
        "sampled_records": len(X_scaled),
        "cluster_sizes": cluster_sizes,
        "cluster_summaries": cluster_summaries,
        "dendrogram_chart": dendrogram_chart,
        "cluster_chart": cluster_chart,
        "features": features
    }

    print("Hierarchical clustering successfully executed.")
    return results


if __name__ == "__main__":
    res = run_hierarchical_clustering()
    print("Execution complete. Sample results:", res["silhouette_score"], res["cophenetic_coeff"])
