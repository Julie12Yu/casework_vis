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

    # Create hover text (brief version for hover)
    hover_text = []
    for d in docs:
        hover = f"{d['name']}\nLow: {d['low_cluster_name']}\nHigh: {d['high_cluster_name']}"
        hover_text.append(hover)

    # Prepare extra_point_data with case names and summaries for on_click
    extra_point_data = pd.DataFrame({
        'case_name': [d['name'] for d in docs],
        'summary': [d['summary'] for d in docs],
        'low_cluster': [d['low_cluster_name'] for d in docs],
        'high_cluster': [d['high_cluster_name'] for d in docs]
    })

    # Custom HTML element to display case details
    custom_html = """
    <div id="case-details" style="
        position: fixed;
        top: 80px;
        right: 20px;
        width: 350px;
        max-height: 70vh;
        background: white;
        border: 2px solid #333;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        overflow-y: auto;
        display: none;
        z-index: 1000;
    ">
        <button onclick="document.getElementById('case-details').style.display='none'" 
                style="float: right; background: #ff4444; color: white; border: none; 
                       padding: 5px 10px; cursor: pointer; border-radius: 4px;">×</button>
        <h3 id="case-name" style="margin-top: 0; color: #333;"></h3>
        <div style="margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px;">
            <div><strong>Low-level cluster:</strong> <span id="low-cluster"></span></div>
            <div><strong>High-level cluster:</strong> <span id="high-cluster"></span></div>
        </div>
        <h4 style="color: #555;">Summary</h4>
        <p id="case-summary" style="line-height: 1.6; color: #666;"></p>
    </div>
    """

    # Custom CSS for dark mode compatibility
    custom_css = """
    #case-details {
        font-family: Arial, sans-serif;
    }
    
    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        #case-details {
            background: #2a2a2a !important;
            border-color: #555 !important;
            color: #e0e0e0 !important;
        }
        #case-name {
            color: #e0e0e0 !important;
        }
        #case-details h4 {
            color: #c0c0c0 !important;
        }
        #case-summary {
            color: #b0b0b0 !important;
        }
        #case-details > div {
            background: #333 !important;
        }
    }
    """

    on_click_js = """
        (function() {{
            var detailsDiv = document.getElementById('case-details');
            var caseName = document.getElementById('case-name');
            var lowCluster = document.getElementById('low-cluster');
            var highCluster = document.getElementById('high-cluster');
            var caseSummary = document.getElementById('case-summary');
            
            function escapeHtml(text) {{
                var div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }}
            
            caseName.textContent = `{case_name}`;
            lowCluster.textContent = `{low_cluster}`;
            highCluster.textContent = `{high_cluster}`;
            caseSummary.textContent = `{summary}`;
            
            detailsDiv.style.display = 'block';
        }})();
    """

    # Use create_interactive_plot for hover text support
    plot = datamapplot.create_interactive_plot(
        embeddings_2d,
        label_names_low,
        label_names_high,
        hover_text=hover_text,
        extra_point_data=extra_point_data,
        title="Court Cases Opinion Landscape",
        on_click=on_click_js,
        custom_html=custom_html,
        custom_css=custom_css,
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