import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

from load_data import load_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def find_eps(X=None, min_samples=5, save_plot=True, filename="dbscan_k_distance.png"):
    """
    Find optimal Epsilon (eps) value for DBSCAN using NearestNeighbors
    and KneeLocator on the sorted K-Distance Graph.
    """
    if X is None:
        # Standalone execution or default RentRadar fallback
        if os.path.exists(os.path.join(BASE_DIR, "preprocessed_placement.csv")):
            data = pd.read_csv(os.path.join(BASE_DIR, "preprocessed_placement.csv"))
            if len(data) > 1000:
                data = data.sample(n=1000, random_state=42)
            if "PlacementStatus" in data.columns:
                X = data.drop("PlacementStatus", axis=1)
            else:
                X = data.copy()
        else:
            data = load_data().copy()
            if len(data) > 1000:
                data = data.sample(n=1000, random_state=42)
            features = ["price", "bedrooms", "bathrooms", "square_feet", "latitude", "longitude"]
            avail = [c for c in features if c in data.columns]
            X = data[avail].copy()
            for c in avail:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            X = X.dropna()

    min_samples = int(min_samples) if min_samples else 5
    if min_samples < 2:
        min_samples = 2

    # Fit Nearest Neighbors with ball_tree algorithm
    neighbors = NearestNeighbors(n_neighbors=min_samples, algorithm='ball_tree')
    neighbors.fit(X)

    distances, indices = neighbors.kneighbors(X)

    # Sort 5th (or min_samples-th) nearest neighbor distances
    k_distances = sorted(distances[:, -1])

    x = range(len(k_distances))

    knee = KneeLocator(
        x,
        k_distances,
        curve="convex",
        direction="increasing"
    )

    knee_idx = knee.knee
    if knee_idx is None or knee_idx >= len(k_distances):
        # Fallback to maximum distance from diagonal line if KneeLocator doesn't converge
        p1 = np.array([0, k_distances[0]])
        p2 = np.array([len(k_distances) - 1, k_distances[-1]])
        dists = []
        for i, val in enumerate(k_distances):
            p3 = np.array([i, val])
            d = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
            dists.append(d)
        knee_idx = int(np.argmax(dists))

    eps = round(float(k_distances[knee_idx]), 4)

    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances, color="#18a6a8", linewidth=2.5, label="k-Distance")
        plt.axvline(x=knee_idx, color="#ef4444", linestyle="--", linewidth=1.8, label=f"Knee Point (Index {knee_idx})")
        plt.axhline(y=eps, color="#3b82f6", linestyle=":", linewidth=1.8, label=f"Optimal Eps = {eps}")
        plt.xlabel("Data Points (Sorted by Distance)", fontsize=11, fontweight="bold")
        plt.ylabel(f"{min_samples}th Nearest Neighbors Distance", fontsize=11, fontweight="bold")
        plt.title(f"K-Distance Graph (DBSCAN Epsilon Detection - Eps: {eps})", fontsize=13, fontweight="bold", pad=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", frameon=True)
        plt.tight_layout()

        chart_path = os.path.join(CHART_DIR, filename)
        plt.savefig(chart_path, dpi=150)
        plt.close()

    return eps


if __name__ == "__main__":
    print("Testing knee_for_DBSCAN standalone...")
    calculated_eps = find_eps()
    print("Calculated Optimal Epsilon (eps):", calculated_eps)
