import json
from collections import defaultdict
from typing import Dict, List, Any

INPUT_FILE_PATH = "hdbscan_results_3d.json"
OUTPUT_FILE_PATH = "categories_from_clusters.json"

def extract_summaries(file_path: str) -> Dict[int, List[str]]:
    """
    Load the 3D embedding JSON and return a dict of {cluster_label: [summaries...]}.
    
    Expected input JSON keys:
      - "points": List[List[float]]  (unused here)
      - "labels": List[int]
      - "titles": List[str]          (unused here)
      - "summaries": List[str]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Basic validation
    required_keys = ["labels", "summaries"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Input file missing required key: '{key}'")

    labels: List[int] = data["labels"]
    summaries: List[str] = data["summaries"]

    if len(labels) != len(summaries):
        raise ValueError(
            f"Length mismatch: labels={len(labels)} vs summaries={len(summaries)}"
        )

    # Group summaries by cluster label
    clusters: Dict[int, List[str]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[int(label)].append(summaries[idx])

    # Convert defaultdict to a normal dict
    return dict(clusters)

if __name__ == "__main__":
    summaries_by_cluster = extract_summaries(INPUT_FILE_PATH)
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as json_file:
        json.dump(summaries_by_cluster, json_file, indent=4, ensure_ascii=False)