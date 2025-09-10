import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Scene + Camera + Renderer
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeeeeee);

const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 0, 20);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

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

// Load JSON embedding data
fetch('3d_embedding.json')
  .then((res) => res.json())
  .then((data) => {
    const { points, labels, summaries } = data;

    points.forEach((p, i) => {
      const [x, y, z = 0] = p;
      const label = labels[i];
      const summary = summaries[i];

      const color = labelColor.get(label);
      const geometry = new THREE.SphereGeometry(0.03, 8, 8);
      const material = new THREE.MeshBasicMaterial({ color });
      const sphere = new THREE.Mesh(geometry, material);

      const scale = 0.6;
      sphere.position.set(x * scale, y * scale, z * scale);

      // Store summary inside the sphere
      sphere.userData.label = label;
      sphere.userData.summary = summary;

      scene.add(sphere);
    });
  });

function onClick(event) {
  // Get mouse coords
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(scene.children);

  if (intersects.length > 0) {
    const obj = intersects[0].object;
    if (obj.userData.summary) {
      summaryDiv.innerText = obj.userData.summary;
      summaryDiv.style.display = 'block';
    }
  }
}

window.addEventListener('click', onClick);

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
