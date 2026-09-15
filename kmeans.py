import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
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
# PERFORMANCE SETTINGS
# =========================================================

# Only this many rows are used when finding the best K.
# This prevents Elbow/Silhouette from becoming extremely slow.
EVALUATION_SAMPLE_SIZE = 5000

RANDOM_STATE = 42


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_clustering_data():

    print("Loading rental dataset...")

    data = load_data().copy()

    print(f"Dataset loaded: {len(data):,} rows")

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

    # Convert everything to numeric
    for col in available_features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # Remove invalid values
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

    print(
        f"Valid clustering records: {len(X):,}"
    )

    # Standardization
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return (
        data.loc[X.index].copy(),
        X,
        X_scaled,
        available_features
    )


# =========================================================
# CREATE EVALUATION SAMPLE
# =========================================================

def create_evaluation_sample(X_scaled):

    sample_size = min(
        EVALUATION_SAMPLE_SIZE,
        len(X_scaled)
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(X_scaled),
        size=sample_size,
        replace=False
    )

    sample = X_scaled[indices]

    print(
        f"Using {sample_size:,} rows for "
        f"K-selection evaluation."
    )

    return sample


# =========================================================
# TRAIN FINAL MODEL
# =========================================================

def train_final_kmeans(
    X_scaled,
    k
):

    print(
        f"Training final MiniBatchKMeans "
        f"with K={k}..."
    )

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=2048,
        n_init=5,
        max_iter=100
    )

    labels = model.fit_predict(
        X_scaled
    )

    print("Final clustering completed.")

    return model, labels


# =========================================================
# ELBOW METHOD
# =========================================================

def calculate_elbow(
    X_sample,
    min_k,
    max_k
):

    print(
        "\nRunning Elbow Method..."
    )

    k_values = list(
        range(
            min_k,
            max_k + 1
        )
    )

    inertias = []

    for k in k_values:

        print(
            f"  Testing K={k}..."
        )

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=2048,
            n_init=3,
            max_iter=80
        )

        model.fit(X_sample)

        inertias.append(
            float(model.inertia_)
        )

    # -----------------------------------------------------
    # AUTOMATIC ELBOW CALCULATION
    # -----------------------------------------------------

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

        if denominator != 0:

            distance = (
                numerator / denominator
            )

        else:

            distance = 0

        distances.append(
            distance
        )

    best_index = int(
        np.argmax(distances)
    )

    suggested_k = k_values[
        best_index
    ]

    # -----------------------------------------------------
    # CREATE CHART
    # -----------------------------------------------------

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

    print(
        f"Elbow Method suggests K={suggested_k}"
    )

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
    X_sample,
    min_k,
    max_k
):

    print(
        "\nRunning Silhouette Method..."
    )

    k_values = list(
        range(
            min_k,
            max_k + 1
        )
    )

    scores = []

    for k in k_values:

        print(
            f"  Testing K={k}..."
        )

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=2048,
            n_init=3,
            max_iter=80
        )

        labels = model.fit_predict(
            X_sample
        )

        # silhouette is calculated ONLY
        # on the 5000-row sample
        score = silhouette_score(
            X_sample,
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

    # -----------------------------------------------------
    # CREATE CHART
    # -----------------------------------------------------

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

    print(
        f"Silhouette Method suggests K={suggested_k}"
    )

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

    print(
        "Creating cluster visualization..."
    )

    # Plot a sample rather than all 100k points.
    # This makes chart generation much faster.
    plot_size = min(
        10000,
        len(data)
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(data),
        size=plot_size,
        replace=False
    )

    plot_data = data.iloc[
        indices
    ]

    plot_labels = np.asarray(
        labels
    )[indices]

    plt.figure(
        figsize=(10, 6)
    )

    scatter = plt.scatter(
        plot_data["square_feet"],
        plot_data["price"],
        c=plot_labels,
        cmap="viridis",
        alpha=0.6,
        s=12
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

    print(
        "Cluster visualization created."
    )

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

    print(
        "\n=========================================="
    )

    print(
        "        K-MEANS CLUSTERING"
    )

    print(
        "=========================================="
    )

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    (
        original_data,
        X,
        X_scaled,
        features
    ) = prepare_clustering_data()

    # -----------------------------------------------------
    # LIMIT K RANGE
    # -----------------------------------------------------

    min_k = max(
        2,
        int(min_k)
    )

    max_k = min(
        15,
        int(max_k)
    )

    if max_k <= min_k:

        max_k = min_k + 1

    # -----------------------------------------------------
    # SELECT K
    # -----------------------------------------------------

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

        selected_k = int(
            manual_k
        )

        selected_k = max(
            2,
            min(selected_k, 15)
        )

        results["selection"] = {

            "type": "Manual K",

            "message": (
                f"K = {selected_k} "
                "was selected manually."
            )
        }

    # -----------------------------------------------------
    # ELBOW
    # -----------------------------------------------------

    elif method == "elbow":

        X_sample = create_evaluation_sample(
            X_scaled
        )

        elbow = calculate_elbow(
            X_sample,
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

        X_sample = create_evaluation_sample(
            X_scaled
        )

        silhouette = calculate_silhouette(
            X_sample,
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
    # FINAL MODEL
    # -----------------------------------------------------

    model, labels = train_final_kmeans(
        X_scaled,
        selected_k
    )

    # -----------------------------------------------------
    # SILHOUETTE OF FINAL MODEL
    # -----------------------------------------------------

    # Calculate this on a sample too.
    # Calculating silhouette on 100k records is expensive.

    evaluation_sample_size = min(
        5000,
        len(X_scaled)
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    sample_indices = rng.choice(
        len(X_scaled),
        size=evaluation_sample_size,
        replace=False
    )

    sample_X = X_scaled[
        sample_indices
    ]

    sample_labels = np.asarray(
        labels
    )[sample_indices]

    final_silhouette = silhouette_score(
        sample_X,
        sample_labels
    )

    # -----------------------------------------------------
    # CLUSTER SIZES
    # -----------------------------------------------------

    cluster_sizes = {}

    unique_clusters, counts = np.unique(
        labels,
        return_counts=True
    )

    for cluster, count in zip(
        unique_clusters,
        counts
    ):

        cluster_sizes[
            int(cluster)
        ] = int(count)

    # -----------------------------------------------------
    # CLUSTER CHART
    # -----------------------------------------------------

    cluster_chart = create_cluster_chart(
        original_data,
        labels,
        selected_k
    )

    # -----------------------------------------------------
    # FINAL RESULTS
    # -----------------------------------------------------

    results.update({

        "selected_k": int(
            selected_k
        ),

        "inertia": round(
            float(model.inertia_),
            2
        ),

        "silhouette_score": round(
            float(final_silhouette),
            4
        ),

        "cluster_sizes": cluster_sizes,

        "cluster_chart": cluster_chart
    })

    print(
        "\n=========================================="
    )

    print(
        "K-MEANS COMPLETED"
    )

    print(
        f"Selected K: {selected_k}"
    )

    print(
        f"Inertia: {model.inertia_:.2f}"
    )

    print(
        f"Silhouette: {final_silhouette:.4f}"
    )

    print(
        "==========================================\n"
    )

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

    print(
        "\n========== K-MEANS TEST =========="
    )

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