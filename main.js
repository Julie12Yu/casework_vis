import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { scrapeCaseName, createUtilityUI, renderLegend} from './utility.js';

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
camera.position.set(11, 15, 5);
window.addEventListener('click', onClick);
window.addEventListener('mousemove', onMouseMove);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);
const ambient = new THREE.AmbientLight(0xffffff, 1.6);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);

// Lighting
const light = new THREE.PointLight(0xffffff, 1);
light.position.set(10, 10, 10);
scene.add(light);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.maxDistance = 20;
controls.minDistance = 6;

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

const groupsByLabel = new Map(); // label -> [{ title, summary, sphere }]
const allSpheres = []; // quick access to every sphere

function getMouse(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

function emphasizeLabel(label) {
  allSpheres.forEach((s) => {
    const match = s.userData.label === label;
    s.material.transparent = true;
    s.material.opacity = match ? 1.0 : 0.15;
  });
}

function createOutline(sphere) {
  const outline = new THREE.Mesh(
    sphere.geometry.clone(),
    new THREE.MeshBasicMaterial({
      color: 0x222222,
      side: THREE.BackSide,
      depthTest: false,
      transparent: true,
      opacity: 0.6
    })
  );
  outline.renderOrder = 999;
  outline.scale.setScalar(1.2);
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

const { summaryDiv, legendDiv, tooltip } = createUtilityUI();
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
      const geometry = new THREE.SphereGeometry(0.03, 32, 32);

      const material = new THREE.MeshStandardMaterial({
        color,
        metalness: 0.0,
        roughness: 0.5,
        transparent: true,
        opacity: 1.0
      });

      const sphere = new THREE.Mesh(geometry, material);

      const scale = 1.1;
      sphere.position.set(x * scale, y * scale, z * scale);

      // Metadata
      sphere.userData.title = title;
      sphere.userData.summary = summary;
      sphere.userData.label = label;

      scene.add(sphere);
      allSpheres.push(sphere);

      if (!groupsByLabel.has(label)) groupsByLabel.set(label, []);
      groupsByLabel.get(label).push({ title, summary, sphere });
    });
    renderLegend(legendDiv, groupsByLabel, labelColor, controls, summaryDiv, allSpheres, clearEmphasis, emphasizeLabel, removeOutline);
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
  const intersects = raycaster.intersectObjects(allSpheres, false);
  if (intersects.length > 0) {
    const obj = intersects[0].object;
    removeOutline();
    if (obj.userData.summary) {
      summaryDiv.innerText = obj.userData.title + "\n------\n" + obj.userData.summary;
      summaryDiv.style.display = 'block';
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
