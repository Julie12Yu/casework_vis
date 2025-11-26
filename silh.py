#!/usr/bin/env python3
"""
Compute silhouette scores from new_court_cases_processed.json

Assumes the JSON was created by your two-level clustering pipeline and
each document has:
  - x, y
  - fine_cluster
  - mid_cluster
  - hdbscan_cluster
"""

import json
import numpy as np
from sklearn.metrics import silhouette_score
from pathlib import Path

INPUT_JSON = "new_court_cases_processed.json"


def load_data(path: str):
    print(f"Loading processed data from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    docs = obj["documents"]
    print(f"  Loaded {len(docs)} documents")

    # 2D coordinates
    X = np.array([[d["x"], d["y"]] for d in docs], dtype=float)

    # Cluster labels
    labels_fine = np.array([d["fine_cluster"] for d in docs], dtype=int)
    labels_mid = np.array([d["mid_cluster"] for d in docs], dtype=int)
    labels_hdbscan = np.array([d["hdbscan_cluster"] for d in docs], dtype=int)

    return X, labels_fine, labels_mid, labels_hdbscan


def safe_silhouette(X, labels, name: str):
    """Compute silhouette score if there are at least 2 clusters."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"  [!] Cannot compute silhouette for {name}: only {len(unique_labels)} cluster(s).")
        return None

    score = silhouette_score(X, labels)
    print(f"  Silhouette score ({name}): {score:.4f}")
    return score


def main():
    if not Path(INPUT_JSON).exists():
        print(f"ERROR: {INPUT_JSON} not found. Update INPUT_JSON or move script.")
        return

    X, labels_fine, labels_mid, labels_hdbscan = load_data(INPUT_JSON)

    print("\nComputing silhouette scores using 2D coordinates (x, y):")

    # 1) Fine K-Means clusters
    sil_fine = safe_silhouette(X, labels_fine, "fine clusters (K-Means)")

    # 2) Mid-level K-Means topics
    sil_mid = safe_silhouette(X, labels_mid, "mid topics (K-Means)")

    # 3) HDBSCAN (ignoring noise)
    print("\nHandling HDBSCAN labels (ignoring noise points with label -1)...")
    mask_core = labels_hdbscan != -1
    n_core = np.sum(mask_core)
    n_total = len(labels_hdbscan)
    print(f"  Core (non-noise) points: {n_core}/{n_total}")

    if n_core > 0:
        X_core = X[mask_core]
        labels_core = labels_hdbscan[mask_core]
        sil_hdb = safe_silhouette(X_core, labels_core, "HDBSCAN core clusters")
    else:
        sil_hdb = None
        print("  [!] No core HDBSCAN points to evaluate.")

    print("\nSummary:")
    print(f"  Fine clusters silhouette:    {sil_fine:.4f}" if sil_fine is not None else "  Fine clusters silhouette:    N/A")
    print(f"  Mid topics silhouette:       {sil_mid:.4f}" if sil_mid is not None else "  Mid topics silhouette:       N/A")
    print(f"  HDBSCAN core silhouette:     {sil_hdb:.4f}" if sil_hdb is not None else "  HDBSCAN core silhouette:     N/A")


if __name__ == "__main__":
    main()
