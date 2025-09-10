import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Scene + Camera + Renderer
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 0, 20); // pull back

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

// Load JSON embedding data
fetch('3d_embedding.json')
  .then((res) => res.json())
  .then((data) => {
    const { points, labels } = data;

    points.forEach((p, i) => {
      const [x, y, z = 0] = p;
      const label = labels[i];

      const color =
        label === -1
          ? 0x888888
          : new THREE.Color(`hsl(${(label * 40) % 360}, 70%, 50%)`);

      const geometry = new THREE.SphereGeometry(0.2, 8, 8);
      const material = new THREE.MeshStandardMaterial({ color });
      const sphere = new THREE.Mesh(geometry, material);

      sphere.position.set(x, y, z);
      scene.add(sphere);
    });
  });


// DON'T TOUCH
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
