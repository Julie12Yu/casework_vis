import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Scene + Camera + Renderer
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeeeeee);
let highlighted = null;

const camera = new THREE.PerspectiveCamera(
  30,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(8, 9, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
window.addEventListener('click', onClick);
window.addEventListener('mousemove', onMouseMove);

// Lighting
const light = new THREE.PointLight(0xffffff, 1);
light.position.set(10, 10, 10);
scene.add(light);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);

// Debug helper
const axesHelper = new THREE.AxesHelper(5);
scene.add(axesHelper);

// Map color to label
const labelColor = new Map();
labelColor.set(-1, 0x808080); // gray for -1

const useableColors = [
  'CornflowerBlue',
  'Crimson',
  'DarkCyan',
  'DarkGreen',
  'LawnGreen',
  'DeepPink',
  'MidnightBlue',
  'OrangeRed'
];
for (let i = 0; i < 8; i++) {
  labelColor.set(i, new THREE.Color(useableColors[i]));
}

// Raycaster + mouse
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

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

const groupsByLabel = new Map(); // label -> [{ title, summary, sphere }]
const allSpheres = []; // quick access to every sphere

function getMouse(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
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

function renderLegend() {
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
      showSummariesForLabel(lbl);
      emphasizeLabel(lbl);
    });

    legendDiv.appendChild(row);
  });

  // Reset button
  const resetHighlightBtn = createButton('Reset highlights position');
  resetHighlightBtn.addEventListener('click', () => {
    clearEmphasis();
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

// Show all summaries for a label in the right-hand panel
function showSummariesForLabel(label) {
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

// Dim non-matching spheres and emphasize matching ones
function emphasizeLabel(label) {
  allSpheres.forEach((s) => {
    const match = s.userData.label === label;
    s.material.transparent = true;
    s.material.opacity = match ? 1.0 : 0.15;
    s.scale.setScalar(match ? 1.8 : 1.0);
  });
}

function createOutline(sphere) {
  const outline = new THREE.Mesh(
    sphere.geometry.clone(),
    new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.BackSide, depthTest: false })
  );
  outline.renderOrder = 999;
  outline.scale.setScalar(1.5);
  sphere.add(outline);
  sphere.userData.outline = outline;
  highlighted = sphere;
}

function removeOutline() {
    if (highlighted?.userData.outline) {
        highlighted.remove(highlighted.userData.outline);
        delete highlighted.userData.outline;
    }
    highlighted = null;
}

// Reset visual emphasis
function clearEmphasis() {
  allSpheres.forEach((s) => {
    s.material.transparent = true;
    s.material.opacity = 1.0;
    s.scale.setScalar(1.0);
  });
}

function scrapeCaseName(rawCaseName) {
  const date = rawCaseName.match(/^\d{4}-\d{2}-\d{2}/);
  const caseText = rawCaseName
    .replace(/\.pdf$/i, '')
    .replace(/^\d{4}-\d{2}-\d{2}_[^_]+_\d{2}-cv-\d+_/, '');
  return `${date} :: ${caseText.replace(/_ et al$/i, '').trim()}`;
}

// Load JSON embedding data
fetch('3d_embedding.json')
  .then((res) => res.json())
  .then((data) => {
    const { points, labels, titles, summaries } = data;

    points.forEach((p, i) => {
      const [x, y, z = 0] = p;
      const label = labels[i];
      const summary = summaries[i];
      const filename = titles[i];

      const title = scrapeCaseName(filename);

      const color = labelColor.get(label);
      const geometry = new THREE.SphereGeometry(0.03, 8, 8);
      // NOTE: enable transparency so we can dim
      const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1.0 });
      const sphere = new THREE.Mesh(geometry, material);

      const scale = 0.6;
      sphere.position.set(x * scale, y * scale, z * scale);

      // Store metadata
      sphere.userData.title = title;
      sphere.userData.summary = summary;
      sphere.userData.label = label;

      scene.add(sphere);
      allSpheres.push(sphere);

      if (!groupsByLabel.has(label)) groupsByLabel.set(label, []);
      groupsByLabel.get(label).push({ title, summary, sphere });
    });

    // Build the legend once data is ready
    renderLegend();
});

function onMouseMove(event) {
  getMouse(event);
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(allSpheres, false);

  if (intersects.length > 0) {
    const obj = intersects[0].object;
    tooltip.style.display = 'block';
    tooltip.textContent = obj.userData.title || '';
    tooltip.style.left = event.clientX + 10 + 'px';
    tooltip.style.top = event.clientY + 10 + 'px';
  } else {
    tooltip.style.display = 'none';
  }
}

function onClick(event) {
  getMouse(event);
  raycaster.setFromCamera(mouse, camera);
  // Only intersect actual spheres for reliability
  const intersects = raycaster.intersectObjects(allSpheres, false);

  if (intersects.length > 0) {
    const obj = intersects[0].object;
    removeOutline();
    if (obj.userData.summary) {
      summaryDiv.innerText = obj.userData.title + "\n------\n" + obj.userData.summary;
      summaryDiv.style.display = 'block';
      // Also emphasize its label for context
      emphasizeLabel(obj.userData.label);
      createOutline(obj);
    }
  }
}

// Render loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// Handle window resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
