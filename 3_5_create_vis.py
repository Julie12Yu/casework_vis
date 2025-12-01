#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import re

import datamapplot

INPUT_JSON = "new_court_cases_processed.json"
OUTPUT_HTML = "index.html" # FOR NOW 


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
    
    # First, remove everything from ELI5 onwards
    # \s* matches any whitespace including newlines, so "word.\n\n4." works correctly
    summary_text = re.split(
        r'\s*\d+\.\s*\*{0,2}ELI5\s+Explanation\*{0,2}\s*:',
        summary_text,
        flags=re.IGNORECASE
    )[0].strip()
    
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
        case_name = match.group(2)
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

        hover = (
            f"{name}\n"
            f"Legal Category: {cat}\n\n"
            f"{summary_text}"
        )
        hover_text.append(hover)

    # Extra data for on_click panel
    extra_point_data = pd.DataFrame({
        "case_name": [extract_case_name(d["name"]) for d in docs],
        "summary": [extract_summary_sections(d["summary"]) for d in docs],
        "legal_category": [d.get("legal_category_name", "Unknown") for d in docs]
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
        
        <!-- History Navigation -->
        <div id="history-nav" style="margin-bottom: 15px; padding-bottom: 15px; border-bottom: 2px solid #ddd;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <button id="prev-case" onclick="navigateHistory(-1)" 
                        style="background: #4CAF50; color: white; border: none; 
                               padding: 8px 15px; cursor: pointer; border-radius: 4px; flex: 1; margin-right: 5px;">
                    ← Previous
                </button>
                <span id="history-position" style="padding: 0 10px; font-weight: bold; color: #666;">-</span>
                <button id="next-case" onclick="navigateHistory(1)" 
                        style="background: #4CAF50; color: white; border: none; 
                               padding: 8px 15px; cursor: pointer; border-radius: 4px; flex: 1; margin-left: 5px;">
                    Next →
                </button>
            </div>
            <button onclick="toggleHistoryList()" 
                    style="width: 100%; background: #2196F3; color: white; border: none; 
                           padding: 8px; cursor: pointer; border-radius: 4px;">
                View History (<span id="history-count">0</span>)
            </button>
        </div>
        
        <!-- History List (collapsible) -->
        <div id="history-list" style="display: none; margin-bottom: 15px; padding: 10px; 
                                       background: #f9f9f9; border-radius: 4px; max-height: 200px; 
                                       overflow-y: auto; border: 1px solid #ddd;">
            <h4 style="margin-top: 0; margin-bottom: 10px; color: #333;">Click History:</h4>
            <div id="history-items"></div>
        </div>
        
        <h3 id="case-name" style="margin-top: 0; color: #333;"></h3>
        <div style="margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px;">
            <div><strong>Legal Category:</strong> <span id="legal-category"></span></div>
        </div>
        <div id="case-summary" style="line-height: 1.6; color: #666; white-space: pre-wrap;"></div>
    </div>
    
    <script>
        // Global history management
        var caseHistory = [];
        var currentHistoryIndex = -1;
        var isNavigating = false;  // Flag to prevent adding to history during navigation
        
        function addToHistory(caseName, legalCategory, summary) {
            if (isNavigating) return;  // Don't add to history when navigating
            
            var newCase = {
                name: caseName,
                category: legalCategory,
                summary: summary
            };
            
            // If we're not at the end of history, truncate everything after current position
            if (currentHistoryIndex < caseHistory.length - 1) {
                caseHistory = caseHistory.slice(0, currentHistoryIndex + 1);
            }
            
            // Add new case to history
            caseHistory.push(newCase);
            currentHistoryIndex = caseHistory.length - 1;
            
            updateHistoryUI();
        }
        
        function navigateHistory(direction) {
            var newIndex = currentHistoryIndex + direction;
            
            if (newIndex < 0 || newIndex >= caseHistory.length) {
                return;  // Can't navigate beyond bounds
            }
            
            currentHistoryIndex = newIndex;
            isNavigating = true;  // Set flag to prevent adding to history
            
            var caseData = caseHistory[currentHistoryIndex];
            displayCase(caseData.name, caseData.category, caseData.summary);
            updateHistoryUI();
            
            isNavigating = false;  // Reset flag
        }
        
        function displayCase(caseName, legalCategory, summary) {
            document.getElementById('case-name').textContent = caseName;
            document.getElementById('legal-category').textContent = legalCategory;
            document.getElementById('case-summary').textContent = summary;
            document.getElementById('case-details').style.display = 'block';
        }
        
        function updateHistoryUI() {
            // Update position indicator
            var positionText = caseHistory.length > 0 
                ? (currentHistoryIndex + 1) + '/' + caseHistory.length 
                : '-';
            document.getElementById('history-position').textContent = positionText;
            
            // Update count
            document.getElementById('history-count').textContent = caseHistory.length;
            
            // Enable/disable navigation buttons
            document.getElementById('prev-case').disabled = currentHistoryIndex <= 0;
            document.getElementById('next-case').disabled = currentHistoryIndex >= caseHistory.length - 1;
            
            // Update button styles for disabled state
            var prevBtn = document.getElementById('prev-case');
            var nextBtn = document.getElementById('next-case');
            
            prevBtn.style.opacity = prevBtn.disabled ? '0.5' : '1';
            prevBtn.style.cursor = prevBtn.disabled ? 'not-allowed' : 'pointer';
            nextBtn.style.opacity = nextBtn.disabled ? '0.5' : '1';
            nextBtn.style.cursor = nextBtn.disabled ? 'not-allowed' : 'pointer';
            
            // Update history list
            updateHistoryList();
        }
        
        function updateHistoryList() {
            var historyItems = document.getElementById('history-items');
            historyItems.innerHTML = '';
            
            if (caseHistory.length === 0) {
                historyItems.innerHTML = '<p style="color: #999; font-style: italic;">No cases viewed yet</p>';
                return;
            }
            
            // Show most recent first
            for (var i = caseHistory.length - 1; i >= 0; i--) {
                var caseData = caseHistory[i];
                var isCurrentCase = i === currentHistoryIndex;
                
                var item = document.createElement('div');
                item.style.cssText = 'padding: 8px; margin-bottom: 5px; border-radius: 4px; cursor: pointer; ' +
                                     'border-left: 3px solid ' + (isCurrentCase ? '#4CAF50' : '#ddd') + '; ' +
                                     'background: ' + (isCurrentCase ? '#e8f5e9' : 'white') + ';';
                
                item.innerHTML = '<strong>' + (i + 1) + '.</strong> ' + caseData.name.substring(0, 50) + 
                                (caseData.name.length > 50 ? '...' : '');
                
                item.onclick = (function(index) {
                    return function() {
                        currentHistoryIndex = index;
                        isNavigating = true;
                        var caseData = caseHistory[index];
                        displayCase(caseData.name, caseData.category, caseData.summary);
                        updateHistoryUI();
                        isNavigating = false;
                    };
                })(i);
                
                historyItems.appendChild(item);
            }
        }
        
        function toggleHistoryList() {
            var historyList = document.getElementById('history-list');
            historyList.style.display = historyList.style.display === 'none' ? 'block' : 'none';
        }
    </script>
    """

    # CSS (light + dark mode)
    custom_css = """
    #case-details {
        font-family: Arial, sans-serif;
    }
    
    #history-list {
        font-size: 14px;
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
        #history-nav {
            border-bottom-color: #555 !important;
        }
        #history-position {
            color: #b0b0b0 !important;
        }
        #history-list {
            background: #333 !important;
            border-color: #555 !important;
        }
        #history-items > div {
            background: #2a2a2a !important;
            border-left-color: #555 !important;
        }
        #history-items > div[style*="background: #e8f5e9"] {
            background: #1b3a1b !important;
        }
    }
    """

    # JS template that datamapplot fills with data from extra_point_data
    on_click_js = """
        (function() {{
            try {{
                var detailsDiv = document.getElementById('case-details');
                var caseName = document.getElementById('case-name');
                var legalCategory = document.getElementById('legal-category');
                var caseSummary = document.getElementById('case-summary');
                
                if (!detailsDiv || !caseName || !legalCategory || !caseSummary) {{
                    console.error('Could not find required elements');
                    return;
                }}
                
                var caseNameText = `{case_name}`;
                var legalCategoryText = `{legal_category}`;
                var summaryText = `{summary}`;
                
                caseName.textContent = caseNameText;
                legalCategory.textContent = legalCategoryText;
                caseSummary.textContent = summaryText;
                
                detailsDiv.style.display = 'block';
                
                // Add to history
                addToHistory(caseNameText, legalCategoryText, summaryText);
            }} catch(e) {{
                console.error('Error in click handler:', e);
            }}
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
    print(f"  - Navigate through clicked cases using Previous/Next buttons")
    print(f"  - View full history list by clicking 'View History' button")


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