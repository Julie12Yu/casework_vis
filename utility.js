// utility.js
import * as THREE from 'three';

function showSummariesForLabel(groupsByLabel, label, summaryDiv) {
  const items = groupsByLabel.get(label) || [];
  if (!items.length) return;

  const html = items
    .slice()
    .sort((a, b) => a.title.localeCompare(b.title))
    .map((it, idx) => {
      return `<div style="margin-bottom:12px;">
        <div style="font-weight:bold">${idx + 1}. ${it.title}</div>
        <div style="white-space:pre-wrap">${(it.summary || '').trim()}</div>
      </div>`;
    })
    .join('<hr style="border:none;border-top:1px solid #eee;margin:8px 0" />');

  summaryDiv.innerHTML = html;
  summaryDiv.style.display = 'block';
}

function createButton(buttonText) {
  const button = document.createElement('button');
  button.textContent = buttonText;
  button.style.marginTop = '10px';
  button.style.width = '100%';
  button.style.padding = '8px';
  button.style.border = '1px solid #bbb';
  button.style.borderRadius = '6px';
  button.style.background = '#f7f7f7';
  button.onmouseenter = () => {
    button.style.background = '#e0e0e0';
  };
  button.onmouseleave = () => {
    button.style.background = '#f7f7f7';
  };
  return button;
}

export function scrapeCaseName(rawCaseName) {
  const date = rawCaseName.match(/^\d{4}-\d{2}-\d{2}/);
  const caseText = rawCaseName
    .replace(/\.pdf$/i, '')
    .replace(/^\d{4}-\d{2}-\d{2}_[^_]+_\d{2}-cv-\d+_/, '');
  return `${date} :: ${caseText.replace(/_ et al$/i, '').trim()}`;
}

export function createUtilityUI() {
  // Create overlay div for summaries
  const summaryDiv = document.createElement('div');
  summaryDiv.style.position = 'absolute';
  summaryDiv.style.top = '10px';
  summaryDiv.style.right = '10px';
  summaryDiv.style.width = '300px';
  summaryDiv.style.maxHeight = '400px';
  summaryDiv.style.overflowY = 'auto';
  summaryDiv.style.backgroundColor = 'rgba(255,255,255,0.9)';
  summaryDiv.style.padding = '10px';
  summaryDiv.style.border = '1px solid #ccc';
  summaryDiv.style.borderRadius = '8px';
  summaryDiv.style.fontFamily = 'sans-serif';
  summaryDiv.style.display = 'none'; // hidden until click
  document.body.appendChild(summaryDiv);

  const legendDiv = document.createElement('div');
  legendDiv.style.position = 'absolute';
  legendDiv.style.top = '10px';
  legendDiv.style.left = '10px';
  legendDiv.style.width = '240px';
  legendDiv.style.maxHeight = '60vh';
  legendDiv.style.overflowY = 'auto';
  legendDiv.style.background = 'rgba(255,255,255,0.9)';
  legendDiv.style.padding = '10px';
  legendDiv.style.border = '1px solid #ccc';
  legendDiv.style.borderRadius = '8px';
  legendDiv.style.fontFamily = 'sans-serif';
  legendDiv.style.display = 'none'; // hidden until data loads
  document.body.appendChild(legendDiv);

  const tooltip = document.createElement('div');
  tooltip.style.position = 'absolute';
  tooltip.style.padding = '4px 8px';
  tooltip.style.background = 'rgba(0,0,0,0.75)';
  tooltip.style.color = '#fff';
  tooltip.style.fontSize = '12px';
  tooltip.style.borderRadius = '4px';
  tooltip.style.pointerEvents = 'none';
  tooltip.style.fontFamily = 'sans-serif';
  tooltip.style.display = 'none';
  document.body.appendChild(tooltip);

  return { summaryDiv, legendDiv, tooltip };
}

export function renderLegend(legendDiv, groupsByLabel, labelColor, controls, summaryDiv, allSpheres, clearEmphasis, emphasizeLabel, removeOutline) {
  legendDiv.innerHTML = '<b>Clusters</b><br/><small>Click any to show all summaries</small><hr/>';
  const labels = [...groupsByLabel.keys()].sort((a, b) => a - b);

  labels.forEach((lbl) => {
    const color = labelColor.get(lbl);
    const hex = (color instanceof THREE.Color) ? `#${color.getHexString()}` : '#808080';
    const items = groupsByLabel.get(lbl) || [];

    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.cursor = 'pointer';
    row.style.padding = '6px 4px';
    row.style.borderRadius = '6px';

    row.onmouseenter = () => { 
      row.style.background = '#e0e0e0'; 
    };
    row.onmouseleave = () => { 
      row.style.background = 'transparent'; 
    };

    const swatch = document.createElement('span');
    swatch.style.display = 'inline-block';
    swatch.style.width = '14px';
    swatch.style.height = '14px';
    swatch.style.border = '1px solid #aaa';
    swatch.style.borderRadius = '3px';
    swatch.style.marginRight = '8px';
    swatch.style.background = hex;

    const labelText = document.createElement('span');
    labelText.textContent = (lbl === -1 ? 'Noise/Unlabeled' : `Cluster ${lbl}`) + `  (${items.length})`;

    row.appendChild(swatch);
    row.appendChild(labelText);

    row.addEventListener('click', () => {
      removeOutline();
      showSummariesForLabel(groupsByLabel, lbl, summaryDiv);
      emphasizeLabel(lbl);
    });

    legendDiv.appendChild(row);
  });

  // Reset button
  const resetHighlightBtn = createButton('Reset highlights');
  resetHighlightBtn.addEventListener('click', () => {
    clearEmphasis(allSpheres);
    removeOutline();
    summaryDiv.style.display = 'none';
  });
  legendDiv.appendChild(document.createElement('hr'));
  legendDiv.appendChild(resetHighlightBtn);

  const resetCameraBtn = createButton('Reset camera position');
  resetCameraBtn.addEventListener('click', () => {
    controls.reset();
  });
  legendDiv.appendChild(resetCameraBtn);

  legendDiv.style.display = 'block';
}