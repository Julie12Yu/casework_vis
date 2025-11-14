#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd

import datamapplot

INPUT_JSON = "court_cases_processed.json"
OUTPUT_HTML = "court_cases_visualization.html"


def load_processed_data(input_json):
    print(f"Loading processed data from {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    docs = obj["documents"]
    meta = obj.get("meta", {})

    return docs, meta


def create_visualization(docs, output_file):
    print("\nCreating visualization...")

    # 2D coordinates
    embeddings_2d = np.array([[d["x"], d["y"]] for d in docs])

    # Primary labels (used for colors / legend)
    label_names_low = [d["low_cluster_name"] for d in docs]
    label_names_high = [d["high_cluster_name"] for d in docs]

    # Create hover text as a list of formatted strings (plain text, not HTML)
    hover_text = []
    for d in docs:
        summary = d["summary"]
        
        # Use plain text with newlines - datamapplot handles the rendering
        hover = f"""{d["name"]}
            Low-level: {d["low_cluster_name"]}
            High-level: {d["high_cluster_name"]}

            Summary: {summary}
        """
        hover_text.append(hover)

    # Use create_interactive_plot for hover text support
    plot = datamapplot.create_interactive_plot(
        embeddings_2d,
        label_names_low,
        label_names_high,
        hover_text=hover_text,
        title="Court Cases Opinion Landscape",
    )

    # Save the interactive plot
    plot.save(output_file)
    print(f"Saved to {output_file}")

def main():
    print("=" * 60)
    print("STEP 3: Visualize Court Cases")
    print("=" * 60)

    docs, _ = load_processed_data(INPUT_JSON)

    create_visualization(docs, OUTPUT_HTML)

if __name__ == "__main__":
    main()