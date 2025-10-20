import json
import numpy as np
import plotly.graph_objects as go
import hdbscan
from sklearn.metrics import silhouette_score

# ---------------------------
# Config
# ---------------------------
INPUT_JSON = "cluster_three_embed.json"
OUTPUT_RESULTS_JSON = "hdbscan_results_3d.json"
OUTPUT_HTML = "hdbscan_3d.html"

MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 5

def load_3d_json(path):
    """
    Load 3D embedding JSON with structure:
    {
      "some file.pdf": [[x,y,z], "summary"],
      ...
    }
    Returns:
      points: (N, 3) float32 ndarray
      titles: list[str] length N
      summaries: list[str] length N
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    titles = []
    points = []
    summaries = []

    for title, value in raw.items():
        try:
            coords, summary = value
            # sanity checks
            if not isinstance(coords, (list, tuple)) or len(coords) != 3:
                raise ValueError("coords must be length-3 list/tuple")
            titles.append(title)
            points.append([float(coords[0]), float(coords[1]), float(coords[2])])
            summaries.append(summary if isinstance(summary, str) else str(summary))
        except Exception as e:
            print(f"Warning: skipping {title} due to parse error: {e}")

    points = np.array(points, dtype=np.float32)
    return points, titles, summaries


def run_hdbscan(points, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(points)
    return labels, clusterer


def compute_silhouette(points, labels):
    """
    Returns silhouette on non-noise points, or None if not computable.
    """
    mask = labels != -1
    if mask.sum() == 0:
        print("All points are noise; silhouette not computable.")
        return None
    unique_clusters = set(labels[mask])
    if len(unique_clusters) < 2:
        print(f"Only {len(unique_clusters)} cluster (ignoring noise); silhouette not computable.")
        return None
    score = silhouette_score(points[mask], labels[mask])
    print(f"Silhouette (non-noise): {score:.4f} on {mask.sum()} points")
    return score


def save_results(points, titles, labels, summaries, path):
    data = {
        "points": points.tolist(),
        "labels": [int(l) for l in labels],
        "titles": titles,
        "summaries": summaries
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved HDBSCAN results to {path}")

# ---------------------------
# Main
# ---------------------------
def main():
    points, titles, summaries = load_3d_json(INPUT_JSON)
    n = len(titles)
    print(f"Loaded {n} items")

    if n == 0:
        raise ValueError("No valid points found in input JSON.")

    min_cluster_size, min_samples = MIN_CLUSTER_SIZE, MIN_SAMPLES

    print("Running HDBSCAN...")
    labels, clusterer = run_hdbscan(points, min_cluster_size, min_samples)

    print("Computing silhouette...")
    sil = compute_silhouette(points, labels)

    print("Saving results JSON...")
    save_results(points, titles, labels, summaries, OUTPUT_RESULTS_JSON)

    print("Done.")

if __name__ == "__main__":
    main()