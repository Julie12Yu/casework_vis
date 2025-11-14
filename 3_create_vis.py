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

    # Create hover text as a list of formatted strings
    hover_text = []
    for d in docs:
        summary = d["summary"][:300] + "..." if len(d["summary"]) > 300 else d["summary"]
        text_preview = d["full_text"][:200] + "..." if len(d["full_text"]) > 200 else d["full_text"]
        
        hover = f"""<b>{d["name"]}</b><br>
Low-level: {d["low_cluster_name"]}<br>
High-level: {d["high_cluster_name"]}<br>
<br>
<b>Summary:</b> {summary}<br>
<br>
<b>Text preview:</b> {text_preview}"""
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