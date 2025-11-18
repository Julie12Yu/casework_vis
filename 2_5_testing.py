import json
import numpy as np
from umap import UMAP
import hdbscan
from sklearn.metrics import silhouette_score

EMBEDDINGS_FILE = "embeddings.npz"

def load_embeddings():
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'summaries': data['summaries'],
        'names': data['names'],
        'full_texts': data['full_texts'],
        'text_lengths': data['text_lengths']
    }

def test_umap_hdbscan(
    embeddings,
    # UMAP parameters
    n_components=2,
    n_neighbors=15,
    min_dist=0.01,
    metric='euclidean',
    # HDBSCAN parameters
    min_cluster_size=10,
    min_samples=10,
    cluster_selection_epsilon=0.0,
    cluster_selection_method='eom'
):
    print(f"  UMAP: n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    print(f"  HDBSCAN: min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}, epsilon={cluster_selection_epsilon}, "
          f"method={cluster_selection_method}")
    
    # Reduce dimensions
    print("\nReducing dimensions...")
    reducer = UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric
    )
    embeddings_reduced = reducer.fit_transform(embeddings)
    
    # Cluster
    print("Clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(embeddings_reduced)
    
    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Silhouette score (only for points not in noise)
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            silhouette = silhouette_score(
                embeddings_reduced[mask], 
                labels[mask]
            )
        else:
            silhouette = -1
    else:
        silhouette = -1
    
    print("\nResults:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
    print(f"  Silhouette score: {silhouette:.4f}")
    
    return {
        'embeddings_reduced': embeddings_reduced,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'params': {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'cluster_selection_epsilon': cluster_selection_epsilon,
            'cluster_selection_method': cluster_selection_method
        }
    }

def main():
    # Load pre-computed embeddings
    data = load_embeddings()
    embeddings = data['embeddings']
    
    print(f"\nLoaded {len(embeddings)} embeddings of shape {embeddings.shape}")
    
    # Example 1: Test a single configuration
    print("\n" + "="*60)
    print("EXAMPLE 1: Testing single configuration")
    print("="*60)
    
    result = test_umap_hdbscan(
        embeddings,
        n_neighbors=15,
        min_dist=0.01,
        min_cluster_size=10,
        min_samples=10
    )

if __name__ == "__main__":
    main()