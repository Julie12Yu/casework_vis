import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from umap import UMAP
from sklearn.cluster import KMeans
from datamapplot import create_plot
from openai import OpenAI
import time

INPUT_FILE = "court_cases_with_summaries.json"
OUTPUT_HTML = "court_cases_visualization.html"

def load_data(input_file):
    """Load documents and summaries from JSON file"""
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} documents")
    
    # Print statistics
    total_chars = sum(d['text_length'] for d in data)
    avg_chars = total_chars / len(data) if data else 0
    print(f"\nStatistics:")
    print(f"  Total documents: {len(data)}")
    print(f"  Total characters in originals: {total_chars:,}")
    print(f"  Average document length: {avg_chars:,.0f} characters")
    
    return data

def load_legalbert_model():
    """Load LegalBERT model and tokenizer"""
    print("\nLoading LegalBERT model...")
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("✓ Model loaded")
    return tokenizer, model

def generate_embeddings(summaries, tokenizer, model):
    """
    Generate embeddings from SUMMARIES (not full text)
    Summaries are designed to fit within LegalBERT's 512 token limit
    """
    print("\nGenerating embeddings from summaries...")
    
    embeddings = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    for i, summary in enumerate(summaries):
        # Embed the SUMMARY (not the full text)
        # The tokenizer will handle truncation if needed, but summaries should fit
        inputs = tokenizer(summary, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(embedding[0])
        
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(summaries)} summaries")
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    return np.array(embeddings)

def reduce_dimensions(embeddings, n_components=2):
    """Reduce embeddings to 2D using UMAP"""
    print(f"\nReducing dimensions with UMAP to {n_components}D...")
    reducer = UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(embeddings)
    print("✓ Dimensionality reduction complete")
    return reduced

def cluster_documents(embeddings_2d, n_clusters_low=50, n_clusters_high=10):
    """
    Two-level hierarchical clustering
    - Low level: many fine-grained clusters
    - High level: fewer broad categories
    """
    print(f"\nClustering into {n_clusters_low} fine-grained clusters...")
    kmeans_low = KMeans(n_clusters=n_clusters_low, random_state=42, n_init=10)
    labels_low = kmeans_low.fit_predict(embeddings_2d)
    
    print(f"Clustering into {n_clusters_high} high-level categories...")
    centroids = kmeans_low.cluster_centers_
    kmeans_high = KMeans(n_clusters=n_clusters_high, random_state=42, n_init=10)
    labels_high = kmeans_high.fit_predict(centroids)
    
    # Map low-level clusters to high-level categories
    high_level_labels = np.array([labels_high[label] for label in labels_low])
    
    print("✓ Clustering complete")
    return labels_low, high_level_labels, kmeans_low.cluster_centers_

def generate_cluster_names(summaries, labels, client, delay=1.0):
    """
    Generate names for clusters using OpenAI
    """
    unique_labels = np.unique(labels)
    cluster_names = {}
    
    print(f"\nGenerating names for {len(unique_labels)} clusters...")
    print("⏱️  Rate limiting: 1 second delay between API calls")
    
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
            
            cluster_names[label] = response.choices[0].message.content.strip()
            print(f"  [{i+1}/{len(unique_labels)}] Cluster {label}: {cluster_names[label]}")
            
            # Rate limiting: wait between requests
            if i < len(unique_labels) - 1:
                time.sleep(delay)
                
        except Exception as e:
            print(f"  ✗ Error naming cluster {label}: {e}")
            cluster_names[label] = f"Cluster {label}"
            time.sleep(delay * 2)
    
    print("✓ Cluster naming complete")
    return cluster_names

def create_visualization(embeddings_2d, labels_low, labels_high, cluster_names_low, cluster_names_high, data, output_file):
    """
    Create interactive visualization using datamapplot
    """
    print(f"\nCreating visualization...")
    
    # Create label names for each document
    label_names_low = [cluster_names_low.get(label, f"Cluster {label}") for label in labels_low]
    label_names_high = [cluster_names_high.get(label, f"Category {label}") for label in labels_high]
    
    # Create hover data
    hover_data = pd.DataFrame({
        'case_name': [d['name'] for d in data],
        'summary': [d['summary'][:300] + "..." if len(d['summary']) > 300 else d['summary'] for d in data],
        'text_preview': [d['full_text'][:200] + "..." for d in data]
    })
    
    # Create the plot
    fig = create_plot(
        embeddings_2d,
        label_names_low,
        label_over=label_names_high,
        hover_data=hover_data,
        title="Court Cases Opinion Landscape",
        sub_title="Interactive visualization of legal opinions clustered by content similarity",
        font_family="Arial"
    )
    
    # Save to HTML
    fig.write_html(output_file)
    print(f"✓ Visualization saved to {output_file}")

def main():
    print("=" * 60)
    print("STEP 2: Analyze and Visualize Court Cases")
    print("=" * 60)
    
    # Load data
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
    labels_low, labels_high, centroids = cluster_documents(embeddings_2d, n_clusters_low, n_clusters_high)
    
    # Generate cluster names
    print("\n[6/6] Generating cluster names with OpenAI...")
    
    with open("otherkey.txt") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)
    
    cluster_names_low = generate_cluster_names(summaries, labels_low, client, delay=1.0)
    cluster_names_high = generate_cluster_names(summaries, labels_high, client, delay=1.0)
    
    # Create visualization
    print("\n" + "=" * 60)
    print("Creating final visualization...")
    create_visualization(embeddings_2d, labels_low, labels_high, cluster_names_low, cluster_names_high, data, OUTPUT_HTML)
    
    print("\n" + "=" * 60)
    print("✓ STEP 2 COMPLETE!")
    print("=" * 60)
    print(f"\nVisualization saved to: {OUTPUT_HTML}")
    print("Open this file in your web browser to explore the court cases landscape!")

if __name__ == "__main__":
    main()