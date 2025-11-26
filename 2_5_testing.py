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
5. Justice and Equity: discrimination, bias, civil rights violations, equal protection issues, equity issues, fairness concerns, or systemic bias, alleged or substantiated discrimination or bias caused by AI, automated systems, or algorithms, or related to AI, automated systems, or algorithims. (e.g., hiring, lending, search).
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
    print("\nReducing dimensions with UMAP...")
    reducer = UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=15,
        min_dist=0.01
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced

def independent_clustering(embeddings_2d, n_docs, n_kmeans=None):
    """
    Run K-Means and HDBSCAN independently, then map HDBSCAN into K-Means.
    Following NeurIPS paper methodology:
    - K-Means: high-level topics (high recall, forces all points into clusters)
    - HDBSCAN: specific subclusters (high precision, tight coherent groups)
    """
    # Auto-calculate K-Means cluster count if not specified
    if n_kmeans is None:
        n_kmeans = min(max(n_docs // 25, 15), 30)
    
    print(f"\n Independent clustering (NeurIPS paper style):")
    print(f"  K-Means: {n_kmeans} high-level topics (coarse layer)")
    print(f"  HDBSCAN: Finding tight subclusters (fine layer)")
    
    # 1. K-MEANS: High-level topics (coarse layer)
    print(f"\n Running K-Means clustering...")
    kmeans = KMeans(n_clusters=n_kmeans, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(embeddings_2d)
    print(f"  Created {n_kmeans} high-level topics")
    
    # 2. HDBSCAN: Specific subclusters (fine layer)
    print(f"\n Running HDBSCAN clustering...")
    hdbscan = HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        cluster_selection_epsilon=0.0,
        metric='euclidean'
    )
    labels_hdbscan = hdbscan.fit_predict(embeddings_2d)
    
    n_hdbscan_clusters = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
    n_noise = np.sum(labels_hdbscan == -1)
    noise_pct = 100 * n_noise / len(labels_hdbscan)
    
    print(f"  HDBSCAN found {n_hdbscan_clusters} tight subclusters")
    print(f"  Noise points: {n_noise} ({noise_pct:.1f}%)")
    
    # 3. Verify HDBSCAN clusters nest within K-Means clusters
    print(f"\n Analyzing nesting structure...")
    hdbscan_to_kmeans = {}
    pure_nesting_count = 0
    
    for hdbscan_id in set(labels_hdbscan):
        if hdbscan_id == -1:
            continue
        
        mask = labels_hdbscan == hdbscan_id
        kmeans_in_hdbscan = labels_kmeans[mask]
        
        # Check if this HDBSCAN cluster falls into a single K-Means cluster
        dominant_kmeans = Counter(kmeans_in_hdbscan).most_common(1)[0][0]
        purity = np.sum(kmeans_in_hdbscan == dominant_kmeans) / len(kmeans_in_hdbscan)
        
        hdbscan_to_kmeans[hdbscan_id] = {
            'dominant_kmeans': int(dominant_kmeans),
            'purity': float(purity),
            'size': int(np.sum(mask))
        }
        
        if purity >= 0.95:  # 95%+ in single K-Means cluster
            pure_nesting_count += 1
    
    if len(hdbscan_to_kmeans) > 0:
        nesting_pct = 100 * pure_nesting_count / len(hdbscan_to_kmeans)
        print(f"  {pure_nesting_count}/{len(hdbscan_to_kmeans)} ({nesting_pct:.1f}%) HDBSCAN clusters nest cleanly in K-Means")
    else:
        print(f"  No HDBSCAN clusters found (all noise)")
    
    # 4. Build hierarchy: K-Means (coarse) contains HDBSCAN (fine)
    kmeans_to_hdbscan = defaultdict(list)
    for hdbscan_id, info in hdbscan_to_kmeans.items():
        kmeans_id = info['dominant_kmeans']
        kmeans_to_hdbscan[kmeans_id].append({
            'hdbscan_id': int(hdbscan_id),
            'size': info['size'],
            'purity': info['purity']
        })
    
    # 5. Analyze noise distribution across K-Means clusters
    noise_mask = labels_hdbscan == -1
    if np.sum(noise_mask) > 0:
        noise_kmeans = labels_kmeans[noise_mask]
        noise_distribution = Counter(noise_kmeans)
        print(f"\n Noise points distributed across K-Means clusters:")
        for kmeans_id in sorted(noise_distribution.keys())[:5]:
            count = noise_distribution[kmeans_id]
            print(f"   K-Means {kmeans_id}: {count} noise points")
        if len(noise_distribution) > 5:
            print(f"   ... and {len(noise_distribution) - 5} more")
    
    return labels_kmeans, labels_hdbscan, hdbscan_to_kmeans, kmeans_to_hdbscan

def classify_kmeans_topics_into_categories(summaries, labels_kmeans, client, delay=0.5):
    """
    Classify each K-Means topic into one of the 8 predefined legal categories.
    Returns mapping of kmeans_label -> (category_num, category_name)
    """
    print("\n Classifying K-Means topics into 8 legal categories...")
    unique_kmeans = np.unique(labels_kmeans)
    kmeans_to_category = {}
    
    for i, kmeans_label in enumerate(unique_kmeans):
        # Get documents in this K-Means topic
        cluster_docs = [summaries[j] for j, l in enumerate(labels_kmeans) if l == kmeans_label]
        
        if len(cluster_docs) == 0:
            kmeans_to_category[int(kmeans_label)] = (8, "Unrelated")
            continue
        
        # Sample up to 20 documents
        sample = "\n\n".join(cluster_docs[:30])
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""You are a legal expert classifying a cluster of court cases into ONE legal category.

Think about how AI, ML, automated systems, or algorithims may have impacted the case. 
Use these definitions carefully and be conservative about assigning AI-related labels. 
If it is not clear that AI, ML, or automated systems are central to the dispute, choose 8 (Unrelated).
Only choose Unrelated (8) if the case doesn't fit categories 1-7.

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
            kmeans_to_category[int(kmeans_label)] = (category_num, category_name)
            
            if (i + 1) % 5 == 0 or (i + 1) == len(unique_kmeans):
                print(f"  [{i+1}/{len(unique_kmeans)}] K-Means Topic {kmeans_label} → {category_name}")
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"  Error classifying K-Means topic {kmeans_label}: {e}")
            kmeans_to_category[int(kmeans_label)] = (8, "Unrelated")
            time.sleep(delay * 2)
    
    # Print distribution
    category_counts = Counter(cat_name for _, cat_name in kmeans_to_category.values())
    print(f"\n Legal category distribution:")
    for cat_name, count in sorted(category_counts.items()):
        print(f"   {cat_name}: {count} topics")
    
    return kmeans_to_category

def generate_hdbscan_subcluster_names(summaries, labels_hdbscan, client, delay=0.5):
    """
    Generate names for HDBSCAN subclusters (tight, specific groups).
    These are the fine-grained, high-precision clusters.
    """
    unique_hdbscan = [l for l in np.unique(labels_hdbscan) if l != -1]
    cluster_names = {}
    
    print(f"  Generating names for {len(unique_hdbscan)} HDBSCAN subclusters...")
    
    for i, hdbscan_label in enumerate(unique_hdbscan):
        cluster_docs = [summaries[j] for j, l in enumerate(labels_hdbscan) if l == hdbscan_label]
        
        if len(cluster_docs) == 0:
            cluster_names[int(hdbscan_label)] = f"Subcluster {hdbscan_label}"
            continue
        
        # Use fewer samples since HDBSCAN clusters are tight and coherent
        sample = "\n\n".join(cluster_docs[:25])
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""You are a legal expert naming a tight, coherent subcluster of court cases.

This is a cluster of similar court cases.

Task:
- Provide ONE concise name (3-7 words) that captures the specific shared theme.
- Be SPECIFIC - this is a tight cluster, not a broad category.
- Focus on the commonality these cases share.

Court case summaries from this cluster:
{sample}

Respond with ONLY the subcluster name. No explanations, quotes, or extra text.

Subcluster name:"""
                }],
                max_tokens=100,
                temperature=0.3
            )
            
            name = response.choices[0].message.content.strip().strip('"\'')
            cluster_names[int(hdbscan_label)] = name
            
            if (i + 1) % 10 == 0 or (i + 1) == len(unique_hdbscan):
                print(f"    [{i+1}/{len(unique_hdbscan)}] Subcluster {hdbscan_label}: {name}")
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"    Error naming subcluster {hdbscan_label}: {e}")
            cluster_names[int(hdbscan_label)] = f"Subcluster {hdbscan_label}"
            time.sleep(delay * 2)
    
    return cluster_names

def generate_kmeans_topic_names(summaries, labels_kmeans, labels_hdbscan, 
                                 kmeans_to_category, hdbscan_names, 
                                 kmeans_to_hdbscan, client, delay=0.8):
    """
    Generate names for K-Means topics using HDBSCAN subcluster names as context.
    Format: "Descriptive Topic Name: Legal Category"
    """
    unique_kmeans = np.unique(labels_kmeans)
    cluster_names = {}
    
    print(f"  Generating category-aware names for {len(unique_kmeans)} K-Means topics...")
    
    for i, kmeans_label in enumerate(unique_kmeans):
        cluster_docs = [summaries[j] for j, l in enumerate(labels_kmeans) if l == kmeans_label]
        
        category_num, category_name = kmeans_to_category.get(int(kmeans_label), (8, "Unrelated"))
        
        if len(cluster_docs) == 0:
            cluster_names[int(kmeans_label)] = f"Topic {kmeans_label}: {category_name}"
            continue
        
        # Get HDBSCAN subclusters in this K-Means topic
        subclusters = kmeans_to_hdbscan.get(int(kmeans_label), [])
        subcluster_names = [
            hdbscan_names.get(sc['hdbscan_id'], f"Subcluster {sc['hdbscan_id']}")
            for sc in subclusters[:10]  # Limit to top 10
        ]
        
        sample_docs = "\n\n".join(cluster_docs[:15])
        
        subcluster_context = ""
        if subcluster_names:
            subcluster_context = f"\n\nThis topic contains these specific subclusters:\n" + "\n".join(f"- {name}" for name in subcluster_names)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""You are a legal expert naming a high-level topic cluster of court cases.

This cluster has been classified as: {category_name}
{subcluster_context}

Task:
- Provide ONE concise name (3-7 words) that captures the overarching theme within this category.
- Format: "Specific Topic: {category_name}"
- Example: "Facial Recognition Biometric Claims: Privacy and Data Protection"
- The first part should be broad enough to encompass the subclusters, the category name stays as-is after the colon.

Court case summaries from this topic:
{sample_docs}

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
            
            cluster_names[int(kmeans_label)] = name
            
            if (i + 1) % 5 == 0 or (i + 1) == len(unique_kmeans):
                print(f"    [{i+1}/{len(unique_kmeans)}] Topic {kmeans_label}: {name}")
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"    Error naming topic {kmeans_label}: {e}")
            cluster_names[int(kmeans_label)] = f"Topic {kmeans_label}: {category_name}"
            time.sleep(delay * 2)
    
    return cluster_names

def save_processed_data(
    embeddings_2d,
    labels_kmeans,
    labels_hdbscan,
    kmeans_to_category,
    hdbscan_to_kmeans,
    kmeans_to_hdbscan,
    cluster_names_kmeans,
    cluster_names_hdbscan,
    data,
    output_json
):
    """Save all clustering results with proper hierarchy"""
    processed = []
    
    for i, d in enumerate(data):
        x, y = float(embeddings_2d[i, 0]), float(embeddings_2d[i, 1])
        kmeans_label = int(labels_kmeans[i])
        hdbscan_label = int(labels_hdbscan[i])
        
        category_num, category_name = kmeans_to_category[int(kmeans_label)]
        
        # Get HDBSCAN info if not noise
        hdbscan_name = None
        hdbscan_purity = None
        if hdbscan_label != -1:
            hdbscan_name = cluster_names_hdbscan.get(hdbscan_label, f"Subcluster {hdbscan_label}")
            hdbscan_info = hdbscan_to_kmeans.get(hdbscan_label, {})
            hdbscan_purity = hdbscan_info.get('purity', None)
        
        processed.append({
            "name": d["name"],
            "summary": d["summary"],
            "full_text": d["full_text"],
            "text_length": d["text_length"],
            "x": x,
            "y": y,
            
            # Coarse layer: K-Means topics
            "kmeans_cluster": kmeans_label,
            "kmeans_cluster_name": cluster_names_kmeans.get(kmeans_label, f"Topic {kmeans_label}"),
            
            # Fine layer: HDBSCAN subclusters
            "hdbscan_cluster": hdbscan_label,
            "hdbscan_cluster_name": hdbscan_name,
            "is_hdbscan_noise": hdbscan_label == -1,
            "hdbscan_nesting_purity": hdbscan_purity,
            
            # Legal category metadata
            "legal_category": category_num,
            "legal_category_name": category_name,
        })
    
    # Calculate distributions
    kmeans_sizes = Counter(labels_kmeans)
    hdbscan_sizes = Counter(l for l in labels_hdbscan if l != -1)
    
    # Build hierarchy: K-Means topics contain HDBSCAN subclusters
    hierarchy = {}
    for kmeans_label in sorted([int(k) for k in np.unique(labels_kmeans)]):
        category_num, category_name = kmeans_to_category[int(kmeans_label)]
        
        # Get HDBSCAN subclusters in this K-Means topic
        subclusters_info = kmeans_to_hdbscan.get(int(kmeans_label), [])
        
        # Count noise points in this K-Means cluster
        noise_in_kmeans = np.sum((labels_kmeans == kmeans_label) & (labels_hdbscan == -1))
        
        hierarchy[int(kmeans_label)] = {
            "name": cluster_names_kmeans.get(int(kmeans_label), f"Topic {kmeans_label}"),
            "size": int(kmeans_sizes[int(kmeans_label)]),
            "legal_category": int(category_num),
            "legal_category_name": str(category_name),
            "n_noise_points": int(noise_in_kmeans),
            "hdbscan_subclusters": [
                {
                    "id": int(sc['hdbscan_id']),
                    "name": cluster_names_hdbscan.get(int(sc['hdbscan_id']), f"Subcluster {sc['hdbscan_id']}"),
                    "size": int(sc['size']),
                    "nesting_purity": float(sc['purity'])
                }
                for sc in sorted(subclusters_info, key=lambda x: x['size'], reverse=True)
            ]
        }
    
    # Category distribution
    category_distribution = {str(k): int(v) for k, v in Counter(
        kmeans_to_category[int(km)][1] for km in np.unique(labels_kmeans)
    ).items()}
    
    output_obj = {
        "documents": processed,
        "meta": {
            "n_documents": int(len(processed)),
            "n_kmeans_topics": int(len(cluster_names_kmeans)),
            "n_hdbscan_subclusters": int(len(cluster_names_hdbscan)),
            "n_hdbscan_noise": int(np.sum(labels_hdbscan == -1)),
            "methodology": "Independent K-Means and HDBSCAN clustering (NeurIPS paper style)",
            
            "cluster_names_kmeans": {int(k): str(v) for k, v in cluster_names_kmeans.items()},
            "cluster_names_hdbscan": {int(k): str(v) for k, v in cluster_names_hdbscan.items()},
            
            "legal_categories": {int(k): str(v) for k, v in CATEGORY_NAMES.items()},
            "category_distribution": category_distribution,
            "kmeans_to_category": {int(k): {"number": int(v[0]), "name": str(v[1])} 
                                   for k, v in kmeans_to_category.items()},
            
            "hierarchy": hierarchy,
        }
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved processed data to {output_json}")

def main():
    print("\n" + "="*60)
    print("INDEPENDENT K-MEANS + HDBSCAN CLUSTERING")
    print("(Following NeurIPS paper methodology)")
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
    
    print(f"\n[4/7] Running independent clustering...")
    print(f"  Dataset size: {len(data)} documents")
    
    (labels_kmeans, labels_hdbscan, 
     hdbscan_to_kmeans, kmeans_to_hdbscan) = independent_clustering(
        embeddings_2d, len(data)
    )
    
    print("\n[5/7] Classifying K-Means topics into legal categories...")
    with open("otherkey.txt") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)
    
    kmeans_to_category = classify_kmeans_topics_into_categories(
        summaries, labels_kmeans, client, delay=0.5
    )
    
    print("\n[6/7] Generating cluster names with OpenAI...")
    
    print("\n Step 1: Naming HDBSCAN subclusters (fine, specific groups)...")
    cluster_names_hdbscan = generate_hdbscan_subcluster_names(
        summaries, labels_hdbscan, client, delay=0.5
    )
    
    print("\n Step 2: Naming K-Means topics (broad, high-level)...")
    cluster_names_kmeans = generate_kmeans_topic_names(
        summaries, labels_kmeans, labels_hdbscan,
        kmeans_to_category, cluster_names_hdbscan,
        kmeans_to_hdbscan, client, delay=0.8
    )
    
    print("\n[7/7] Saving processed data...")
    save_processed_data(
        embeddings_2d,
        labels_kmeans,
        labels_hdbscan,
        kmeans_to_category,
        hdbscan_to_kmeans,
        kmeans_to_hdbscan,
        cluster_names_kmeans,
        cluster_names_hdbscan,
        data,
        OUTPUT_JSON
    )
    
    print("\n" + "="*60)
    print("CLUSTERING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_JSON}")
    print(f"\nMethodology (NeurIPS paper style):")
    print(f"  - K-Means: {len(cluster_names_kmeans)} high-level topics (coarse layer)")
    print(f"  - HDBSCAN: {len(cluster_names_hdbscan)} specific subclusters (fine layer)")
    print(f"  - Noise: {np.sum(labels_hdbscan == -1)} points ({100*np.sum(labels_hdbscan == -1)/len(labels_hdbscan):.1f}%)")
    print(f"\nHierarchy:")
    print(f"  - Coarse: K-Means topics with legal categories")
    print(f"  - Fine: HDBSCAN subclusters nested within K-Means topics")
    print(f"\nKey insight:")
    print(f"  - HDBSCAN subclusters have high precision (tight, specific)")
    print(f"  - K-Means topics have high recall (broad, catch everything)")
    print(f"  - HDBSCAN names inform K-Means topic names")

if __name__ == "__main__":
    main()