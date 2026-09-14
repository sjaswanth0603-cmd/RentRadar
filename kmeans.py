import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from load_data import load_data


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)


# =========================================================
# FEATURES USED FOR CLUSTERING
# =========================================================

CLUSTER_FEATURES = [
    "price",
    "bedrooms",
    "bathrooms",
    "square_feet",
    "latitude",
    "longitude"
]


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_clustering_data():

    data = load_data().copy()

    if "_dataset_source" in data.columns:
        data = data.drop(
            columns=["_dataset_source"]
        )

    available_features = [
        col
        for col in CLUSTER_FEATURES
        if col in data.columns
    ]

    if len(available_features) < 2:
        raise ValueError(
            "Not enough numerical features available "
            "for K-Means clustering."
        )

    X = data[
        available_features
    ].copy()

    for col in available_features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.dropna()

    if len(X) < 20:
        raise ValueError(
            "Not enough valid records available "
            "for K-Means clustering."
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return (
        data.loc[X.index].copy(),
        X,
        X_scaled,
        available_features
    )


# =========================================================
# MANUAL K
# =========================================================

def run_manual_k(
    X_scaled,
    original_data,
    features,
    k
):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    result_data = original_data.copy()

    result_data["cluster"] = labels

    cluster_sizes = (
        result_data["cluster"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "method": "Manual K",
        "selected_k": k,
        "inertia": round(
            float(model.inertia_),
            2
        ),
        "silhouette_score": round(
            float(silhouette),
            4
        ),
        "cluster_sizes": cluster_sizes,
        "features": features,
        "labels": labels,
        "centers": model.cluster_centers_,
        "data": result_data
    }


# =========================================================
# ELBOW METHOD
# =========================================================

def calculate_elbow(
    X_scaled,
    min_k,
    max_k
):

    k_values = list(
        range(
            min_k,
            max_k + 1
        )
    )

    inertias = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        inertias.append(
            float(model.inertia_)
        )

    # Simple automatic elbow calculation
    points = np.column_stack(
        (
            k_values,
            inertias
        )
    )

    first = points[0]
    last = points[-1]

    distances = []

    for point in points:

        numerator = abs(
            (last[1] - first[1]) * point[0]
            - (last[0] - first[0]) * point[1]
            + last[0] * first[1]
            - last[1] * first[0]
        )

        denominator = np.sqrt(
            (last[1] - first[1]) ** 2
            + (last[0] - first[0]) ** 2
        )

        distance = (
            numerator / denominator
            if denominator != 0
            else 0
        )

        distances.append(
            distance
        )

    best_index = int(
        np.argmax(distances)
    )

    suggested_k = k_values[
        best_index
    ]

    chart_path = os.path.join(
        CHART_DIR,
        "kmeans_elbow.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        k_values,
        inertias,
        marker="o"
    )

    plt.axvline(
        suggested_k,
        linestyle="--",
        label=f"Suggested K = {suggested_k}"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "WCSS / Inertia"
    )

    plt.title(
        "K-Means Elbow Method"
    )

    plt.xticks(
        k_values
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        chart_path,
        dpi=150
    )

    plt.close()

    return {
        "k_values": k_values,
        "inertias": inertias,
        "suggested_k": suggested_k,
        "chart": "charts/kmeans_elbow.png"
    }


# =========================================================
# SILHOUETTE METHOD
# =========================================================

def calculate_silhouette(
    X_scaled,
    min_k,
    max_k
):

    k_values = list(
        range(
            min_k,
            max_k + 1
        )
    )

    scores = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels
        )

        scores.append(
            float(score)
        )

    best_index = int(
        np.argmax(scores)
    )

    suggested_k = k_values[
        best_index
    ]

    chart_path = os.path.join(
        CHART_DIR,
        "kmeans_silhouette.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        k_values,
        scores,
        marker="o"
    )

    plt.axvline(
        suggested_k,
        linestyle="--",
        label=f"Best K = {suggested_k}"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        "K-Means Silhouette Method"
    )

    plt.xticks(
        k_values
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        chart_path,
        dpi=150
    )

    plt.close()

    return {
        "k_values": k_values,
        "scores": scores,
        "suggested_k": suggested_k,
        "chart": "charts/kmeans_silhouette.png"
    }


# =========================================================
# CLUSTER VISUALIZATION
# =========================================================

def create_cluster_chart(
    data,
    labels,
    selected_k
):

    chart_path = os.path.join(
        CHART_DIR,
        "kmeans_clusters.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    scatter = plt.scatter(
        data["square_feet"],
        data["price"],
        c=labels,
        cmap="viridis",
        alpha=0.6
    )

    plt.xlabel(
        "Square Feet"
    )

    plt.ylabel(
        "Rental Price"
    )

    plt.title(
        f"K-Means Clusters (K = {selected_k})"
    )

    plt.colorbar(
        scatter,
        label="Cluster"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        chart_path,
        dpi=150
    )

    plt.close()

    return "charts/kmeans_clusters.png"


# =========================================================
# MAIN K-MEANS FUNCTION
# =========================================================

def run_kmeans(
    method="manual",
    manual_k=3,
    min_k=2,
    max_k=10
):

    (
        original_data,
        X,
        X_scaled,
        features
    ) = prepare_clustering_data()

    results = {
        "method": method,
        "features": features,
        "total_records": len(X),
        "selected_k": None,
        "inertia": None,
        "silhouette_score": None,
        "cluster_sizes": {},
        "selection": None,
        "cluster_chart": None
    }

    # -----------------------------------------------------
    # MANUAL
    # -----------------------------------------------------

    if method == "manual":

        selected_k = manual_k

        results["selection"] = {
            "type": "Manual K",
            "message": (
                f"K = {selected_k} was selected manually."
            )
        }

    # -----------------------------------------------------
    # ELBOW
    # -----------------------------------------------------

    elif method == "elbow":

        elbow = calculate_elbow(
            X_scaled,
            min_k,
            max_k
        )

        selected_k = elbow[
            "suggested_k"
        ]

        results["selection"] = {
            "type": "Elbow Method",
            "suggested_k": selected_k,
            "chart": elbow["chart"],
            "k_values": elbow["k_values"],
            "inertias": elbow["inertias"]
        }

    # -----------------------------------------------------
    # SILHOUETTE
    # -----------------------------------------------------

    elif method == "silhouette":

        silhouette = calculate_silhouette(
            X_scaled,
            min_k,
            max_k
        )

        selected_k = silhouette[
            "suggested_k"
        ]

        results["selection"] = {
            "type": "Silhouette Method",
            "suggested_k": selected_k,
            "chart": silhouette["chart"],
            "k_values": silhouette["k_values"],
            "scores": silhouette["scores"]
        }

    else:

        raise ValueError(
            "Invalid K selection method."
        )

    # -----------------------------------------------------
    # FINAL K-MEANS MODEL
    # -----------------------------------------------------

    clustering = run_manual_k(
        X_scaled=X_scaled,
        original_data=original_data,
        features=features,
        k=selected_k
    )

    cluster_chart = create_cluster_chart(
        original_data,
        clustering["labels"],
        selected_k
    )

    results.update({
        "selected_k": clustering["selected_k"],
        "inertia": clustering["inertia"],
        "silhouette_score": clustering["silhouette_score"],
        "cluster_sizes": clustering["cluster_sizes"],
        "cluster_chart": cluster_chart
    })

    return results


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = run_kmeans(
        method="silhouette",
        min_k=2,
        max_k=10
    )

    print("\n========== K-MEANS TEST ==========")

    print(
        "Method:",
        result["method"]
    )

    print(
        "Selected K:",
        result["selected_k"]
    )

    print(
        "Silhouette Score:",
        result["silhouette_score"]
    )

    print(
        "Cluster Sizes:",
        result["cluster_sizes"]
    )