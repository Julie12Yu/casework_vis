import * as THREE from 'three';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';

export class CartographicLayer {
  constructor(scene, groupsByLabel, labelColor) {
    this.scene = scene;
    this.groupsByLabel = groupsByLabel;
    this.labelColor = labelColor;
    this.countryMeshes = [];
    this.borderLines = [];
    this.contourMeshes = [];
  }

  create() {
    this.groupsByLabel.forEach((items, label) => {
      if (label === -1) return; // Skip noise points
      
      const points = items.map(item => item.sphere.position.clone());
      if (points.length < 4) return; // Need at least 4 points for ConvexGeometry
      
      const color = this.labelColor.get(label);
      
      // 1. Create filled region
      const region = this.createRegion(points, color);
      if (region) {
        this.scene.add(region);
        this.countryMeshes.push(region);
      }
      
      // 2. Create border
      const border = this.createBorder(points, color);
      if (border) {
        this.scene.add(border);
        this.borderLines.push(border);
      }
      
      // 3. Create contours
      const center = this.computeCenter(points);
      const contours = this.createContours(center, points, color);
      contours.forEach(c => {
        this.scene.add(c);
        this.contourMeshes.push(c);
      });
    });
  }

  createRegion(points, color) {
    try {
      const geometry = new ConvexGeometry(points);
      const material = new THREE.MeshPhongMaterial({
        color: color,
        transparent: true,
        opacity: 0.15,
        side: THREE.DoubleSide,
        flatShading: true,
        depthWrite: false
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = -1; // Render behind points
      return mesh;
    } catch (e) {
      console.warn('Could not create region:', e);
      return null;
    }
  }

  createBorder(points, color) {
    try {
      const geometry = new ConvexGeometry(points);
      const edges = new THREE.EdgesGeometry(geometry);
      const line = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ 
          color: new THREE.Color(color).multiplyScalar(0.7), // Slightly darker
          transparent: true,
          opacity: 0.6,
          linewidth: 2
        })
      );
      return line;
    } catch (e) {
      console.warn('Could not create border:', e);
      return null;
    }
  }

  computeCenter(points) {
    const center = new THREE.Vector3();
    points.forEach(p => center.add(p));
    center.divideScalar(points.length);
    return center;
  }

  createContours(center, points, color) {
    // Compute average distance for scaling
    const avgDist = points.reduce((sum, p) => 
      sum + p.distanceTo(center), 0) / points.length;
    
    const contours = [];
    const levels = [0.6, 0.8, 1.0];
    
    levels.forEach((scale, i) => {
      const geometry = new THREE.IcosahedronGeometry(
        avgDist * scale, 1
      );
      const material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(color).multiplyScalar(0.8),
        transparent: true,
        opacity: 0.08 * (levels.length - i),
        wireframe: true,
        depthWrite: false
      });
      const contour = new THREE.Mesh(geometry, material);
      contour.position.copy(center);
      contour.renderOrder = -2;
      contours.push(contour);
    });
    
    return contours;
  }

  setOpacity(opacity) {
    this.countryMeshes.forEach(mesh => {
      mesh.material.opacity = opacity * 0.15;
    });
    this.borderLines.forEach(line => {
      line.material.opacity = opacity * 0.6;
    });
    this.contourMeshes.forEach(mesh => {
      mesh.material.opacity = opacity * 0.08;
    });
  }

  setVisible(visible) {
    [...this.countryMeshes, ...this.borderLines, ...this.contourMeshes].forEach(obj => {
      obj.visible = visible;
    });
  }

  destroy() {
    [...this.countryMeshes, ...this.borderLines, ...this.contourMeshes].forEach(obj => {
      this.scene.remove(obj);
      obj.geometry?.dispose();
      obj.material?.dispose();
    });
    this.countryMeshes = [];
    this.borderLines = [];
    this.contourMeshes = [];
  }
}