#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import re

import datamapplot

INPUT_JSON = "new_court_cases_processed.json"
OUTPUT_HTML = "court_cases_visualization.html"


def load_processed_data(input_json):
    print(f"Loading processed data from {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    docs = obj["documents"]
    meta = obj.get("meta", {})

    print(f"  Loaded {len(docs)} documents")
    return docs, meta


def extract_summary_sections(summary_text):
    """Extract only Summary and Key Legal Issue sections from the summary."""
    summary_section = ""
    legal_issue_section = ""
    
    # Summary section
    summary_match = re.search(
        r'\d+\.\s*\*{0,2}Summary\*{0,2}\s*:\s*(.+?)(?=\n\s*\d+\.\s*\*{0,2}(?:Key Legal Issue|Court|ELI5)|\Z)',
        summary_text,
        re.DOTALL | re.IGNORECASE
    )
    if summary_match:
        summary_section = summary_match.group(1).strip()
    
    # Key Legal Issue section
    legal_issue_match = re.search(
        r'\d+\.\s*\*{0,2}Key Legal Issue\*{0,2}\s*:\s*(.+?)(?=\n\s*\d+\.\s*\*{0,2}(?:Summary|Court|ELI5)|\Z)',
        summary_text,
        re.DOTALL | re.IGNORECASE
    )
    if legal_issue_match:
        legal_issue_section = legal_issue_match.group(1).strip()
    
    # Combine nicely
    result = ""
    if summary_section:
        result += f"Summary: {summary_section}\n\n"
    if legal_issue_section:
        result += f"Key Legal Issue: {legal_issue_section}"
    
    return result.strip() if result else summary_text


def extract_case_name(filename):
    """
    Convert filenames like '2024-05-01_XYZ_v_ABC_Some-Case-Name.pdf'
    into 'Some-Case-Name (2024-05-01)'.
    Falls back to the raw filename if pattern doesn't match.
    """
    pattern = r'^(\d{4}-\d{2}-\d{2})_[^_]+_[^_]+_(.+?)\.pdf$'
    match = re.match(pattern, filename)
    
    if match:
        date = match.group(1)
        case_name = match.group(2).rstrip('.')
        return f"{case_name} ({date})"
    
    return filename


def create_visualization(docs, output_file):
    print("\nCreating visualization...")

    # 2D coordinates from processed JSON
    embeddings_2d = np.array([[d["x"], d["y"]] for d in docs])

    # Primary labels used by datamapplot (colors / legend)
    # Fine-grained clusters = "low"; mid-level topics = "high"
    label_names_low = [d["fine_cluster_name"] for d in docs]
    label_names_high = [d["mid_cluster_name"] for d in docs]

    # Hover text
    hover_text = []
    for d in docs:
        name = extract_case_name(d["name"])
        summary_text = extract_summary_sections(d["summary"])
        cat = d.get("legal_category_name", "Unknown")
        quality = d.get("cluster_quality", "unknown")

        hover = (
            f"{name}\n"
            f"Fine cluster: {d['fine_cluster_name']} [{quality}]\n"
            f"Mid topic: {d['mid_cluster_name']}\n"
            f"Legal category: {cat}\n\n"
            f"{summary_text}"
        )
        hover_text.append(hover)

    # Extra data for on_click panel
    extra_point_data = pd.DataFrame({
        "case_name": [extract_case_name(d["name"]) for d in docs],
        "summary": [d["summary"] for d in docs],
        "fine_cluster": [d["fine_cluster_name"] for d in docs],
        "mid_cluster": [d["mid_cluster_name"] for d in docs],
        "legal_category": [d.get("legal_category_name", "Unknown") for d in docs],
        "cluster_quality": [d.get("cluster_quality", "unknown") for d in docs],
    })

    # Right-hand details panel (HTML injected into the page)
    custom_html = """
    <div id="case-details" style="
        position: fixed;
        top: 80px;
        right: 20px;
        width: 400px;
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
            <div><strong>Fine cluster:</strong> <span id="fine-cluster"></span></div>
            <div><strong>Mid topic:</strong> <span id="mid-cluster"></span></div>
            <div><strong>Legal category:</strong> <span id="legal-category"></span></div>
            <div><strong>Cluster quality:</strong> <span id="cluster-quality"></span></div>
        </div>
        <div id="case-summary" style="line-height: 1.6; color: #666; white-space: pre-wrap;"></div>
    </div>
    """

    # CSS (light + dark mode)
    custom_css = """
    #case-details {
        font-family: Arial, sans-serif;
    }
    
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

    # JS template that datamapplot fills with data from extra_point_data
    # Placeholders {case_name}, {fine_cluster}, etc. must match the DataFrame columns
    on_click_js = """
        (function() {{
            var detailsDiv = document.getElementById('case-details');
            var caseName = document.getElementById('case-name');
            var fineCluster = document.getElementById('fine-cluster');
            var midCluster = document.getElementById('mid-cluster');
            var legalCategory = document.getElementById('legal-category');
            var clusterQuality = document.getElementById('cluster-quality');
            var caseSummary = document.getElementById('case-summary');
            
            caseName.textContent = "{case_name}";
            fineCluster.textContent = "{fine_cluster}";
            midCluster.textContent = "{mid_cluster}";
            legalCategory.textContent = "{legal_category}";
            clusterQuality.textContent = "{cluster_quality}";
            caseSummary.textContent = "{summary}";
            
            detailsDiv.style.display = 'block';
        }})();
    """


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

    plot.save(output_file)
    print(f"Saved visualization to {output_file}")


def main():
    print("=" * 60)
    print("STEP 3: Visualize Court Cases (Fine + Mid + Legal Categories)")
    print("=" * 60)

    docs, _ = load_processed_data(INPUT_JSON)
    create_visualization(docs, OUTPUT_HTML)


if __name__ == "__main__":
    main()
