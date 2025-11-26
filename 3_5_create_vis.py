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
    print(f"  Methodology: {meta.get('methodology', 'Unknown')}")
    print(f"  K-Means topics: {meta.get('n_kmeans_topics', '?')}")
    print(f"  HDBSCAN subclusters: {meta.get('n_hdbscan_subclusters', '?')}")
    print(f"  HDBSCAN noise points: {meta.get('n_hdbscan_noise', '?')}")
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


def create_visualization(docs, meta, output_file):
    print("\nCreating visualization...")

    # 2D coordinates from processed JSON
    embeddings_2d = np.array([[d["x"], d["y"]] for d in docs])

    # NEW: Use K-Means (coarse) and HDBSCAN (fine) labels
    # K-Means = high-level topics (coarse layer, always assigned)
    # HDBSCAN = specific subclusters (fine layer, may be noise)
    label_names_high = [d["kmeans_cluster_name"] for d in docs]
    
    # For HDBSCAN, show subcluster name or "Unclustered" for noise
    label_names_low = []
    for d in docs:
        if d["is_hdbscan_noise"]:
            # Noise points - show their K-Means topic + "Unclustered"
            label_names_low.append(f"{d['kmeans_cluster_name']} (Unclustered)")
        else:
            label_names_low.append(d["hdbscan_cluster_name"])

    # Hover text
    hover_text = []
    for d in docs:
        name = extract_case_name(d["name"])
        summary_text = extract_summary_sections(d["summary"])
        cat = d.get("legal_category_name", "Unknown")
        
        # HDBSCAN status
        if d["is_hdbscan_noise"]:
            hdbscan_info = "Unclustered (noise)"
            purity_info = ""
        else:
            hdbscan_info = d["hdbscan_cluster_name"]
            purity = d.get("hdbscan_nesting_purity")
            if purity is not None:
                purity_info = f" [purity: {purity:.2f}]"
            else:
                purity_info = ""

        hover = (
            f"{name}\n"
            f"K-Means Topic: {d['kmeans_cluster_name']}\n"
            f"HDBSCAN Subcluster: {hdbscan_info}{purity_info}\n"
            f"Legal Category: {cat}\n\n"
            f"{summary_text}"
        )
        hover_text.append(hover)

    # Extra data for on_click panel
    extra_point_data = pd.DataFrame({
        "case_name": [extract_case_name(d["name"]) for d in docs],
        "summary": [d["summary"] for d in docs],
        "kmeans_cluster": [d["kmeans_cluster_name"] for d in docs],
        "hdbscan_cluster": [
            "Unclustered (noise)" if d["is_hdbscan_noise"] 
            else d["hdbscan_cluster_name"] 
            for d in docs
        ],
        "legal_category": [d.get("legal_category_name", "Unknown") for d in docs],
        "hdbscan_status": [
            "🔍 High-precision subcluster" if not d["is_hdbscan_noise"] 
            else "📊 Unclustered (in K-Means topic)" 
            for d in docs
        ],
        "nesting_purity": [
            f"{d.get('hdbscan_nesting_purity', 0):.1%}" if not d["is_hdbscan_noise"]
            else "N/A"
            for d in docs
        ]
    })

    # Right-hand details panel (HTML injected into the page)
    custom_html = """
    <div id="case-details" style="
        position: fixed;
        top: 80px;
        right: 20px;
        width: 450px;
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
        <div style="margin: 10px 0; padding: 15px; background: #f5f5f5; border-radius: 4px;">
            <div style="margin-bottom: 8px;"><strong>📂 K-Means Topic:</strong> <span id="kmeans-cluster"></span></div>
            <div style="margin-bottom: 8px;"><strong>🔬 HDBSCAN Subcluster:</strong> <span id="hdbscan-cluster"></span></div>
            <div style="margin-bottom: 8px;"><strong>⚖️ Legal Category:</strong> <span id="legal-category"></span></div>
            <div style="margin-bottom: 8px;"><strong>📊 Status:</strong> <span id="hdbscan-status"></span></div>
            <div><strong>🎯 Nesting Purity:</strong> <span id="nesting-purity"></span></div>
        </div>
        <h4 style="margin-top: 15px; color: #555;">Case Summary</h4>
        <div id="case-summary" style="line-height: 1.6; color: #666; white-space: pre-wrap;"></div>
    </div>
    """

    # CSS (light + dark mode)
    custom_css = """
    #case-details {
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    #case-details h3 {
        font-size: 18px;
        border-bottom: 2px solid #007bff;
        padding-bottom: 8px;
    }
    
    #case-details h4 {
        font-size: 14px;
        color: #555;
        margin-top: 15px;
    }
    
    @media (prefers-color-scheme: dark) {
        #case-details {
            background: #2a2a2a !important;
            border-color: #555 !important;
            color: #e0e0e0 !important;
        }
        #case-name {
            color: #e0e0e0 !important;
            border-bottom-color: #4a9eff !important;
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
    on_click_js = """
        (function() {{
            var detailsDiv = document.getElementById('case-details');
            var caseName = document.getElementById('case-name');
            var kmeansCluster = document.getElementById('kmeans-cluster');
            var hdbscanCluster = document.getElementById('hdbscan-cluster');
            var legalCategory = document.getElementById('legal-category');
            var hdbscanStatus = document.getElementById('hdbscan-status');
            var nestingPurity = document.getElementById('nesting-purity');
            var caseSummary = document.getElementById('case-summary');
            
            caseName.textContent = "{case_name}";
            kmeansCluster.textContent = "{kmeans_cluster}";
            hdbscanCluster.textContent = "{hdbscan_cluster}";
            legalCategory.textContent = "{legal_category}";
            hdbscanStatus.textContent = "{hdbscan_status}";
            nestingPurity.textContent = "{nesting_purity}";
            caseSummary.textContent = "{summary}";
            
            detailsDiv.style.display = 'block';
        }})();
    """

    # Create plot
    plot = datamapplot.create_interactive_plot(
        embeddings_2d,
        label_names_low,      # HDBSCAN subclusters (fine detail)
        label_names_high,     # K-Means topics (broad categories)
        hover_text=hover_text,
        extra_point_data=extra_point_data,
        title="Court Cases Landscape",
        on_click=on_click_js,
        custom_html=custom_html,
        custom_css=custom_css,
    )

    plot.save(output_file)
    print(f"✓ Saved visualization to {output_file}")
    print(f"\nVisualization notes:")
    print(f"  - Colors/regions: K-Means topics (high-level)")
    print(f"  - Labels when zoomed: HDBSCAN subclusters (specific)")
    print(f"  - Noise points: Shown as 'Unclustered' within their K-Means topic")
    print(f"  - Click any point to see full details in right panel")


def main():
    print("=" * 60)
    print("STEP 3: Visualize Court Cases")
    print("Independent K-Means + HDBSCAN Clustering (NeurIPS Style)")
    print("=" * 60)

    docs, meta = load_processed_data(INPUT_JSON)
    create_visualization(docs, meta, OUTPUT_HTML)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()