import json
import numpy as np
import torch
import time

from transformers import AutoTokenizer, AutoModel
from umap import UMAP
from sklearn.cluster import KMeans
from openai import OpenAI

INPUT_FILE = "court_cases_with_summaries.json"
OUTPUT_JSON = "court_cases_processed.json"


def load_data(input_file):
    """Load documents and summaries from JSON file"""
    print(f"Loading data from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def load_legalbert_model():
    """Load LegalBERT model and tokenizer"""
    print("\nLoading LegalBERT model...")
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def generate_embeddings(summaries, tokenizer, model):

    embeddings = []
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    for i, summary in enumerate(summaries):
        inputs = tokenizer(
            summary,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        embeddings.append(embedding[0])

    return np.array(embeddings)


def reduce_dimensions(embeddings, n_components=2):
    reducer = UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


def cluster_documents(embeddings_2d, n_clusters_low=50, n_clusters_high=10):
    kmeans_low = KMeans(n_clusters=n_clusters_low, random_state=42, n_init=10)
    labels_low = kmeans_low.fit_predict(embeddings_2d)

    centroids = kmeans_low.cluster_centers_
    kmeans_high = KMeans(n_clusters=n_clusters_high, random_state=42, n_init=10)
    labels_high_centroids = kmeans_high.fit_predict(centroids)

    # Map low-level clusters to high-level categories
    high_level_labels = np.array([labels_high_centroids[label] for label in labels_low])

    return labels_low, high_level_labels, kmeans_low.cluster_centers_


def generate_cluster_names(summaries, labels, client, delay=1.0):
    """
    Generate names for clusters using OpenAI
    """
    unique_labels = np.unique(labels)
    cluster_names = {}

    for i, label in enumerate(unique_labels):
        # Get summaries for this cluster
        cluster_docs = [summaries[j] for j, l in enumerate(labels) if l == label]

        # Sample up to 5 documents from the cluster
        sample = "\n\n".join(cluster_docs[:5])

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""Given these court case summaries from the same cluster, provide a concise cluster name (2-5 words) that captures the common theme:

{sample}

Cluster name:"""
                }],
                max_tokens=100
            )

            name = response.choices[0].message.content.strip()
            cluster_names[int(label)] = name
            print(f"  [{i+1}/{len(unique_labels)}] Cluster {label}: {name}")

            # Rate limiting: wait between requests
            if i < len(unique_labels) - 1:
                time.sleep(delay)

        except Exception as e:
            print(f"Error naming cluster {label}: {e}")
            cluster_names[int(label)] = f"Cluster {label}"
            time.sleep(delay * 2)

    print("Cluster naming done")
    return cluster_names


def save_processed_data(
    embeddings_2d,
    labels_low,
    labels_high,
    cluster_names_low,
    cluster_names_high,
    data,
    output_json
):
    processed = []
    for i, d in enumerate(data):
        x, y = float(embeddings_2d[i, 0]), float(embeddings_2d[i, 1])
        low_label = int(labels_low[i])
        high_label = int(labels_high[i])

        processed.append({
            "name": d["name"],
            "summary": d["summary"],
            "full_text": d["full_text"],
            "text_length": d["text_length"],
            "x": x,
            "y": y,
            "low_cluster": low_label,
            "high_cluster": high_label,
            "low_cluster_name": cluster_names_low.get(
                low_label,
                f"Cluster {low_label}"
            ),
            "high_cluster_name": cluster_names_high.get(
                high_label,
                f"Category {high_label}"
            ),
        })

    # You can also include cluster name dictionaries at top-level if you want
    output_obj = {
        "documents": processed,
        "meta": {
            "n_documents": len(processed),
            "cluster_names_low": cluster_names_low,
            "cluster_names_high": cluster_names_high,
        }
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)


def main():
    print("\n[1/6] Loading data...")
    data = load_data(INPUT_FILE)

    if not data:
        print("No data found. Please run step1_generate_summaries.py first.")
        return

    # Extract summaries
    summaries = [d['summary'] for d in data]

    # Load LegalBERT
    print("\n[2/6] Loading LegalBERT model...")
    tokenizer, model = load_legalbert_model()

    # Generate embeddings FROM SUMMARIES
    print("\n[3/6] Generating embeddings...")
    embeddings = generate_embeddings(summaries, tokenizer, model)

    # Reduce dimensions
    print("\n[4/6] Reducing dimensions...")
    embeddings_2d = reduce_dimensions(embeddings)

    # Determine cluster counts based on dataset size
    n_docs = len(data)
    n_clusters_low = min(max(n_docs // 10, 10), 50)
    n_clusters_high = min(max(n_docs // 50, 5), 10)

    print(f"\nDataset size: {n_docs} documents")
    print(f"Using {n_clusters_low} fine-grained clusters and {n_clusters_high} high-level categories")

    # Cluster
    print("\n[5/6] Clustering...")
    labels_low, labels_high, _ = cluster_documents(
        embeddings_2d,
        n_clusters_low=n_clusters_low,
        n_clusters_high=n_clusters_high
    )

    # Generate cluster names
    print("\n[6/6] Generating cluster names with OpenAI...")
    with open("otherkey.txt") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)

    cluster_names_low = generate_cluster_names(summaries, labels_low, client, delay=1.0)
    cluster_names_high = generate_cluster_names(summaries, labels_high, client, delay=1.0)

    # Save everything to JSON for later visualization
    print("\n" + "=" * 60)
    print("Saving processed data...")
    save_processed_data(
        embeddings_2d,
        labels_low,
        labels_high,
        cluster_names_low,
        cluster_names_high,
        data,
        OUTPUT_JSON
    )

if __name__ == "__main__":
    main()