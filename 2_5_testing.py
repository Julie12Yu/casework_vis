import json
import numpy as np
import time

from umap import UMAP
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from openai import OpenAI
from collections import defaultdict, Counter

INPUT_FILE = "court_cases_with_summaries.json"
EMBEDDINGS_FILE = "embeddings.npz"
OUTPUT_JSON = "new_court_cases_processed.json"

# Predefined legal categories for high-level classification
LEGAL_CATEGORIES = """
1. Antitrust: market competition, monopolization, market power, anti-competitive practices, market dominance, price-fixing, exclusive dealing, or restraint of trade involving ANY tech companies, or anti-competitive practices by major platforms or AI companies.
2. IP Law: patents, copyrights, trademarks for AI models or tech, or training data disputes, AI-generated content ownership.
3. Privacy and Data Protection: data breaches, unauthorized data collection by automated systems, or privacy violations involving algorithms or data processing.
4. Tort: physical harm, emotional distress, negligence, defamation, or personal injury involving ANY automated systems, tech systems, major tech corporations using AI, or algorithms.
5. Justice and Equity: discrimination, bias, civil rights violations, equal protection issues, or fairness concerns, employment discrimination, housing discrimination, lending discrimination, educational equity, voting rights, and systemic bias, alleged or substantiated discrimination or bias caused by AI, automated systems, or algorithms, or related to AI, automated systems, or algorithims. (e.g., hiring, lending, search).
6. Consumer Protection: deceptive practices, unfair business practices with tech/automated systems, or misleading marketing of tech products or AI capabilities.
7. AI in Legal Proceedings: AI systems are merely used in the court processes, legal case management, or litigation tools. The core contention is not about AI, but AI tools have been used in the litigation process.
8. Unrelated: cases that have no meaningful connection to AI, ML, or automated systems. If the case involves discrimination, privacy, or other issues **without automation/AI/algorithmic involvement**, classify as Unrelated.
"""

CATEGORY_NAMES = {
    1: "Antitrust",
    2: "IP Law",
    3: "Privacy and Data Protection",
    4: "Tort",
    5: "Justice and Equity",
    6: "Consumer Protection",
    7: "AI in Legal Proceedings",
    8: "Unrelated"
}


def load_data(input_file):
    """Load documents and summaries from JSON file"""
    print(f"Loading data from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def load_embeddings(embeddings_file):
    """Load pre-computed embeddings from NPZ file"""
    print(f"\nLoading embeddings from {embeddings_file}...")
    
    with np.load(embeddings_file) as data:
        embeddings = data['embeddings']
    
    print(f"  Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    return embeddings


def reduce_dimensions(embeddings, n_components=2):
    """Reduce embeddings to 2D using UMAP"""
    reducer = UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=15,
        min_dist=0.01
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


def hybrid_clustering_with_quality(embeddings_2d, n_docs, n_fine=None, n_mid=None):
    """
    Hybrid approach with predefined top-level categories:
    
    1. K-Means creates fine and mid-level clusters
    2. Mid-level clusters will be classified into 8 predefined legal categories
    3. HDBSCAN provides quality/confidence scores for fine clusters
    4. Every point gets a K-Means label (no missing names when hovering)
    """
    # Auto-calculate cluster counts
    if n_fine is None:
        n_fine = min(max(n_docs // 8, 40), 100)
    if n_mid is None:
        n_mid = min(max(n_docs // 25, 15), 30)
    
    print(f"\n  Hybrid clustering with quality assessment:")
    print(f"    K-Means Level 1 (Fine): {n_fine} clusters")
    print(f"    K-Means Level 2 (Mid): {n_mid} topics")
    print(f"    Level 3 (High): 8 predefined legal categories (to be classified)")
    print(f"    HDBSCAN: Quality assessment")
    
    # K-MEANS: Two-level hierarchy
    print(f"\n  Running K-Means Level 1: {n_fine} fine-grained clusters...")
    kmeans_fine = KMeans(n_clusters=n_fine, random_state=42, n_init=10)
    labels_fine = kmeans_fine.fit_predict(embeddings_2d)
    centroids_fine = kmeans_fine.cluster_centers_
    
    print(f"  Running K-Means Level 2: {n_mid} mid-level topics...")
    kmeans_mid = KMeans(n_clusters=n_mid, random_state=42, n_init=10)
    labels_mid_centroids = kmeans_mid.fit_predict(centroids_fine)
    centroids_mid = kmeans_mid.cluster_centers_
    labels_mid = np.array([labels_mid_centroids[label] for label in labels_fine])
    
    # HDBSCAN: Quality assessment (finds high-confidence subclusters)
    print(f"\n  Running HDBSCAN for quality assessment...")
    hdbscan = HDBSCAN(
        min_cluster_size=8,
        min_samples=5,
        cluster_selection_epsilon=0.0,
        metric='euclidean'
    )
    hdbscan_labels = hdbscan.fit_predict(embeddings_2d)
    
    n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    n_hdbscan_noise = np.sum(hdbscan_labels == -1)
    
    print(f"    HDBSCAN found {n_hdbscan_clusters} high-confidence subclusters")
    print(f"    HDBSCAN noise: {n_hdbscan_noise} points ({100*n_hdbscan_noise/len(hdbscan_labels):.1f}%)")
    
    # Analyze cluster quality using HDBSCAN
    cluster_quality = assess_cluster_quality(labels_fine, hdbscan_labels, kmeans_fine)
    
    return (labels_fine, labels_mid, 
            hdbscan_labels, cluster_quality)


def assess_cluster_quality(kmeans_labels, hdbscan_labels, kmeans_model):
    """
    Assess quality of K-Means clusters using HDBSCAN results:
    - High quality: K-Means cluster contains one or more HDBSCAN clusters
    - Medium quality: K-Means cluster is mix of HDBSCAN clusters and noise
    - Low quality: K-Means cluster is mostly HDBSCAN noise
    """
    cluster_quality = {}
    
    for k_label in np.unique(kmeans_labels):
        mask = kmeans_labels == k_label
        points_in_cluster = np.sum(mask)
        
        # Get HDBSCAN labels for points in this K-Means cluster
        hdbscan_in_kmeans = hdbscan_labels[mask]
        
        # Count HDBSCAN noise vs clustered points
        n_noise = np.sum(hdbscan_in_kmeans == -1)
        n_clustered = np.sum(hdbscan_in_kmeans != -1)
        
        noise_ratio = n_noise / points_in_cluster
        
        # Get unique HDBSCAN clusters in this K-Means cluster
        hdbscan_clusters_present = set(hdbscan_in_kmeans[hdbscan_in_kmeans != -1])
        n_hdbscan_subclusters = len(hdbscan_clusters_present)
        
        # Calculate inertia (compactness) for this cluster
        cluster_points = np.where(mask)[0]
        centroid = kmeans_model.cluster_centers_[k_label]
        
        # Assess quality
        if noise_ratio < 0.3 and n_hdbscan_subclusters >= 1:
            quality = "high"
            quality_score = 1.0
        elif noise_ratio < 0.6:
            quality = "medium"
            quality_score = 0.5
        else:
            quality = "low"
            quality_score = 0.2
        
        cluster_quality[int(k_label)] = {
            'quality': quality,
            'quality_score': quality_score,
            'noise_ratio': noise_ratio,
            'n_hdbscan_subclusters': n_hdbscan_subclusters,
            'size': points_in_cluster
        }
    
    # Print quality distribution
    quality_counts = Counter(q['quality'] for q in cluster_quality.values())
    print(f"\n  Cluster quality distribution:")
    print(f"    High quality: {quality_counts['high']} clusters (tight, coherent)")
    print(f"    Medium quality: {quality_counts['medium']} clusters (moderate coherence)")
    print(f"    Low quality: {quality_counts['low']} clusters (loose groupings)")
    
    return cluster_quality


def analyze_hierarchy(labels_fine, labels_mid):
    """Analyze how fine clusters map to mid topics"""
    print("\n  Analyzing hierarchy mapping...")
    
    fine_to_mid = defaultdict(set)
    for fine, mid in zip(labels_fine, labels_mid):
        fine_to_mid[fine].add(mid)
    
    fine_pure = sum(1 for mids in fine_to_mid.values() if len(mids) == 1)
    
    print(f"    Fine→Mid purity: {fine_pure}/{len(fine_to_mid)} ({100*fine_pure/len(fine_to_mid):.1f}%)")
    
    return fine_to_mid


def classify_mid_topics_into_categories(summaries, labels_mid, client, delay=0.5):
    """
    Classify each mid-level topic into one of the 8 predefined legal categories.
    This is used to inform the naming, not create a separate hierarchy level.
    Returns mapping of mid_label -> (category_num, category_name)
    """
    print("\n  Classifying mid-level topics into 8 legal categories...")
    
    unique_mid = np.unique(labels_mid)
    mid_to_category = {}
    
    for i, mid_label in enumerate(unique_mid):
        # Get documents in this mid topic
        cluster_docs = [summaries[j] for j, l in enumerate(labels_mid) if l == mid_label]
        
        if len(cluster_docs) == 0:
            mid_to_category[int(mid_label)] = (8, "Unrelated")
            continue
        
        # Sample up to 20 documents
        sample = "\n\n".join(cluster_docs[:20])
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""You are a legal expert classifying a cluster of court cases into ONE legal category.

Think about how AI, ML, automated systems, or algorithims may have impacted the case. 
Use these definitions carefully and be conservative about assigning AI-related labels. 
If it is not clear that AI, ML, or automated systems are central to the dispute, choose 8 (Unrelated).
Only choose Unrelated (8) if the case truly doesn't fit categories 1-7.

{LEGAL_CATEGORIES}

Cases in this cluster:
{sample}

Respond with EXACTLY one digit from 1 to 8 and nothing else.
Category number (1–8):"""

                }],
                max_tokens=10,
                temperature=0
            )
            
            category_num = int(response.choices[0].message.content.strip())
            category_name = CATEGORY_NAMES.get(category_num, "Unknown")
            
            mid_to_category[int(mid_label)] = (category_num, category_name)
            
            if (i + 1) % 5 == 0 or (i + 1) == len(unique_mid):
                print(f"    [{i+1}/{len(unique_mid)}] Mid Topic {mid_label} → {category_name}")
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"    Error classifying mid topic {mid_label}: {e}")
            mid_to_category[int(mid_label)] = (8, "Unrelated")
            time.sleep(delay * 2)
    
    # Print distribution
    category_counts = Counter(cat_name for _, cat_name in mid_to_category.values())
    print(f"\n  Legal category distribution:")
    for cat_name, count in sorted(category_counts.items()):
        print(f"    {cat_name}: {count} mid topics")
    
    return mid_to_category


def generate_category_aware_mid_names(summaries, labels_mid, mid_to_category, client, delay=0.8):
    """
    Generate names for mid-level topics that incorporate the legal category.
    Format: "Descriptive Topic Name: Legal Category"
    """
    unique_mid = np.unique(labels_mid)
    cluster_names = {}

    print(f"  Generating category-aware names for {len(unique_mid)} mid topics...")

    for i, mid_label in enumerate(unique_mid):
        cluster_docs = [summaries[j] for j, l in enumerate(labels_mid) if l == mid_label]
        
        if len(cluster_docs) == 0:
            category_num, category_name = mid_to_category.get(int(mid_label), (8, "Unrelated"))
            cluster_names[int(mid_label)] = f"Topic {mid_label}: {category_name}"
            continue

        sample = "\n\n".join(cluster_docs[:20])
        category_num, category_name = mid_to_category.get(int(mid_label), (8, "Unrelated"))

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""You are a legal expert naming a cluster of court cases.

This cluster has been classified as: {category_name}

Task:
- Provide ONE concise name (3-7 words) that captures the specific theme within this category.
- Format: "Specific Topic: {category_name}"
- Example: "Facial Recognition Biometric Claims: Privacy and Data Protection"
- The first part should be specific, the category name stays as-is after the colon.

Court case summaries from this cluster:
{sample}

Respond with ONLY the cluster name in the format "Topic: Category". No explanations.
Cluster name:"""

                }],
                max_tokens=100,
                temperature=0.3
            )

            name = response.choices[0].message.content.strip().strip('"\'')
            # Ensure format is correct
            if ": " not in name:
                name = f"{name}: {category_name}"
            cluster_names[int(mid_label)] = name
            
            if (i + 1) % 5 == 0 or (i + 1) == len(unique_mid):
                print(f"    [{i+1}/{len(unique_mid)}] Topic {mid_label}: {name}")

            time.sleep(delay)

        except Exception as e:
            print(f"    Error naming topic {mid_label}: {e}")
            cluster_names[int(mid_label)] = f"Topic {mid_label}: {category_name}"
            time.sleep(delay * 2)

    return cluster_names


def generate_cluster_names(summaries, labels, client, cluster_quality=None, 
                          level_name="Cluster", delay=0.8, max_samples=20):
    """
    Generate names for clusters using OpenAI.
    If cluster_quality provided, can adjust naming strategy based on quality.
    """
    unique_labels = np.unique(labels)
    cluster_names = {}

    print(f"  Generating names for {len(unique_labels)} {level_name} clusters...")

    for i, label in enumerate(unique_labels):
        cluster_docs = [summaries[j] for j, l in enumerate(labels) if l == label]
        
        if len(cluster_docs) == 0:
            cluster_names[int(label)] = f"{level_name} {label}"
            continue

        sample_size = min(len(cluster_docs), max_samples)
        sample = "\n\n".join(cluster_docs[:sample_size])
        
        # Adjust prompt based on cluster quality
        quality_note = ""
        if cluster_quality and int(label) in cluster_quality:
            qual = cluster_quality[int(label)]['quality']
            if qual == "high":
                quality_note = " This is a tight, coherent cluster - be specific."
            elif qual == "low":
                quality_note = " This is a loose grouping - focus on the common thread."

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""You are a legal expert naming a cluster of court cases.

Task:
- Provide ONE concise cluster name (2-6 words) that captures the common theme.
- The name should be specific and descriptive.{quality_note}

Court case summaries from this cluster ({len(cluster_docs)} cases):
{sample}

Respond with ONLY the cluster name. Do not include any explanations, quotes, or extra text.
Cluster name:"""

                }],
                max_tokens=100,
                temperature=0.3
            )

            name = response.choices[0].message.content.strip().strip('"\'')
            cluster_names[int(label)] = name
            
            if (i + 1) % 10 == 0 or (i + 1) == len(unique_labels):
                quality_marker = ""
                if cluster_quality and int(label) in cluster_quality:
                    q = cluster_quality[int(label)]['quality']
                    quality_marker = f" [{q[0].upper()}]"
                print(f"    [{i+1}/{len(unique_labels)}]{quality_marker} {level_name} {label}: {name}")

            if i < len(unique_labels) - 1:
                time.sleep(delay)

        except Exception as e:
            print(f"    Error naming cluster {label}: {e}")
            cluster_names[int(label)] = f"{level_name} {label}"
            time.sleep(delay * 2)

    return cluster_names


def save_processed_data(
    embeddings_2d,
    labels_fine,
    labels_mid,
    mid_to_category,
    hdbscan_labels,
    cluster_quality,
    cluster_names_fine,
    cluster_names_mid,
    fine_to_mid,
    data,
    output_json
):
    """Save all clustering results with two-level hierarchy and category metadata"""
    processed = []
    
    for i, d in enumerate(data):
        x, y = float(embeddings_2d[i, 0]), float(embeddings_2d[i, 1])
        fine_label = int(labels_fine[i])
        mid_label = int(labels_mid[i])
        category_num, category_name = mid_to_category[int(mid_label)]
        hdbscan_label = int(hdbscan_labels[i])
        
        # Get quality info for this point's fine cluster
        quality_info = cluster_quality.get(int(fine_label), {})
        
        processed.append({
            "name": d["name"],
            "summary": d["summary"],
            "full_text": d["full_text"],
            "text_length": d["text_length"],
            "x": x,
            "y": y,
            # Two-level hierarchy (fine clusters within mid topics)
            "fine_cluster": fine_label,
            "fine_cluster_name": cluster_names_fine.get(fine_label, f"Cluster {fine_label}"),
            "mid_cluster": mid_label,
            "mid_cluster_name": cluster_names_mid.get(mid_label, f"Topic {mid_label}"),
            # Legal category metadata (not a separate hierarchy level)
            "legal_category": category_num,
            "legal_category_name": category_name,
            # Quality assessment from HDBSCAN
            "cluster_quality": quality_info.get('quality', 'unknown'),
            "cluster_quality_score": quality_info.get('quality_score', 0.5),
            "hdbscan_cluster": hdbscan_label,
            "is_hdbscan_core": hdbscan_label != -1,
        })

    # Calculate distributions
    fine_sizes = Counter(labels_fine)
    mid_sizes = Counter(labels_mid)
    
    # Create two-level hierarchy (mid topics contain fine clusters)
    hierarchy = {}
    
    for mid_label in sorted([int(m) for m in np.unique(labels_mid)]):
        category_num, category_name = mid_to_category[int(mid_label)]
        
        # Get fine clusters in this mid topic
        fine_in_mid = [int(fine) for fine, mids in fine_to_mid.items() if int(mid_label) in mids]
        
        hierarchy[int(mid_label)] = {
            "name": cluster_names_mid.get(int(mid_label), f"Topic {mid_label}"),
            "size": int(mid_sizes[int(mid_label)]),
            "legal_category": int(category_num),
            "legal_category_name": str(category_name),
            "fine_clusters": [
                {
                    "id": int(fine_label),
                    "name": cluster_names_fine.get(int(fine_label), f"Cluster {fine_label}"),
                    "size": int(fine_sizes[int(fine_label)]),
                    "quality": str(cluster_quality.get(int(fine_label), {}).get('quality', 'unknown')),
                    "quality_score": float(cluster_quality.get(int(fine_label), {}).get('quality_score', 0.5))
                }
                for fine_label in sorted(fine_in_mid)
            ]
        }

    # Category distribution for reference
    category_distribution = {str(k): int(v) for k, v in Counter(
        mid_to_category[int(mid)][1] for mid in np.unique(labels_mid)
    ).items()}

    output_obj = {
        "documents": processed,
        "meta": {
            "n_documents": int(len(processed)),
            "n_fine_clusters": int(len(cluster_names_fine)),
            "n_mid_topics": int(len(cluster_names_mid)),
            "cluster_names_fine": {int(k): str(v) for k, v in cluster_names_fine.items()},
            "cluster_names_mid": {int(k): str(v) for k, v in cluster_names_mid.items()},
            "legal_categories": {int(k): str(v) for k, v in CATEGORY_NAMES.items()},
            "category_distribution": category_distribution,
            "mid_to_category": {int(k): {"number": int(v[0]), "name": str(v[1])} for k, v in mid_to_category.items()},
            "cluster_quality": {int(k): {
                "quality": str(v.get('quality', 'unknown')),
                "quality_score": float(v.get('quality_score', 0.5)),
                "noise_ratio": float(v.get('noise_ratio', 0.0)),
                "n_hdbscan_subclusters": int(v.get('n_hdbscan_subclusters', 0)),
                "size": int(v.get('size', 0))
            } for k, v in cluster_quality.items()},
            "hierarchy": hierarchy,
        }
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved processed data to {output_json}")


def main():
    print("\n" + "="*60)
    print("TWO-LEVEL CLUSTERING WITH LEGAL CATEGORY NAMING")
    print("(K-Means hierarchy + HDBSCAN quality + Category-aware names)")
    print("="*60)
    
    print("\n[1/7] Loading data...")
    data = load_data(INPUT_FILE)

    if not data:
        print("No data found. Please run step1_generate_summaries.py first.")
        return

    summaries = [d['summary'] for d in data]

    print("\n[2/7] Loading pre-computed embeddings...")
    embeddings = load_embeddings(EMBEDDINGS_FILE)
    
    if len(embeddings) != len(data):
        print(f"ERROR: Mismatch between embeddings ({len(embeddings)}) and data ({len(data)})")
        return

    print("\n[3/7] Reducing dimensions...")
    embeddings_2d = reduce_dimensions(embeddings)

    print(f"\n[4/7] Hybrid clustering with quality assessment...")
    print(f"  Dataset size: {len(data)} documents")
    
    (labels_fine, labels_mid, 
     hdbscan_labels, cluster_quality) = hybrid_clustering_with_quality(
        embeddings_2d, len(data)
    )
    
    print("\n[5/7] Analyzing hierarchy...")
    fine_to_mid = analyze_hierarchy(labels_fine, labels_mid)

    print("\n[6/7] Generating cluster names with OpenAI...")
    with open("otherkey.txt") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)

    # First classify mid topics into legal categories
    print("\n  Step 1: Classifying mid topics into legal categories...")
    mid_to_category = classify_mid_topics_into_categories(
        summaries, labels_mid, client, delay=0.5
    )

    # Then generate category-aware names for mid topics
    print("\n  Step 2: Generating category-aware names for mid topics...")
    cluster_names_mid = generate_category_aware_mid_names(
        summaries, labels_mid, mid_to_category, client, delay=0.8
    )
    
    print("\n  Step 3: Naming fine-grained clusters...")
    cluster_names_fine = generate_cluster_names(
        summaries, labels_fine, client,
        cluster_quality=cluster_quality,
        level_name="Cluster", delay=0.5, max_samples=10
    )

    print("\n[7/7] Saving processed data...")
    save_processed_data(
        embeddings_2d,
        labels_fine,
        labels_mid,
        mid_to_category,
        hdbscan_labels,
        cluster_quality,
        cluster_names_fine,
        cluster_names_mid,
        fine_to_mid,
        data,
        OUTPUT_JSON
    )
    
    print("\n" + "="*60)
    print("CLUSTERING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_JSON}")
    print(f"  - Level 2 (Mid): {len(cluster_names_mid)} topics with legal categories in names")
    print(f"  - Level 1 (Fine): {len(cluster_names_fine)} specific clusters (zoom detail)")
    print(f"\nHDBSCAN quality assessment included:")
    print(f"  - 'high' quality = tight, coherent clusters")
    print(f"  - 'medium' quality = moderate coherence")
    print(f"  - 'low' quality = loose groupings")
    print(f"\nMid-level topic names include legal categories:")
    print(f"  Example: 'Facial Recognition Claims: Privacy and Data Protection'")
    print(f"\nVisualization zoom behavior:")
    print(f"  - Zoomed out: See {len(cluster_names_mid)} mid topic names (with categories)")
    print(f"  - Zoomed in: See {len(cluster_names_fine)} fine cluster names (specific details)")


if __name__ == "__main__":
    main()