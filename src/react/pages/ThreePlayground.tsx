import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/**
 * Self-contained WebGL scene: starfield, iridescent torus knot, orbit controls.
 * Disposes all GPU resources on unmount.
 */
export function ThreePlayground() {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x09090b, 0.038);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 120);
    camera.position.set(0, 0.4, 5.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setClearColor(0x09090b, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);

    const hemi = new THREE.HemisphereLight(0xffe4c4, 0x09090b, 0.72);
    scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 1.35);
    key.position.set(4, 6, 5);
    key.castShadow = true;
    key.shadow.mapSize.setScalar(1024);
    scene.add(key);

    const rim = new THREE.PointLight(0xfb923c, 38, 24, 2);
    rim.position.set(-2.5, 1.2, 2);
    scene.add(rim);

    const knotGeo = new THREE.TorusKnotGeometry(1.05, 0.32, 220, 36, 2, 3);
    const knotMat = new THREE.MeshPhysicalMaterial({
      color: 0xfb923c,
      metalness: 0.88,
      roughness: 0.16,
      emissive: 0x7c2d12,
      emissiveIntensity: 0.35,
      clearcoat: 1,
      clearcoatRoughness: 0.12,
      iridescence: 1,
      iridescenceIOR: 1.55,
      iridescenceThicknessRange: [80, 280],
    });
    const knot = new THREE.Mesh(knotGeo, knotMat);
    knot.castShadow = true;
    knot.receiveShadow = true;
    scene.add(knot);

    const wireGeo = new THREE.IcosahedronGeometry(2.15, 2);
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0xfdba74,
      wireframe: true,
      transparent: true,
      opacity: 0.12,
    });
    const wire = new THREE.Mesh(wireGeo, wireMat);
    scene.add(wire);

    const starCount = 4200;
    const starPos = new Float32Array(starCount * 3);
    const starCol = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const r = 6 + Math.random() * 22;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);
      starPos[i * 3] = x;
      starPos[i * 3 + 1] = y;
      starPos[i * 3 + 2] = z;
      const t = Math.random();
      starCol[i * 3] = 1;
      starCol[i * 3 + 1] = 0.55 + t * 0.35;
      starCol[i * 3 + 2] = 0.35 + t * 0.25;
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    starGeo.setAttribute("color", new THREE.BufferAttribute(starCol, 3));
    const starMat = new THREE.PointsMaterial({
      size: 0.035,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const stars = new THREE.Points(starGeo, starMat);
    scene.add(stars);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 2.8;
    controls.maxDistance = 14;
    controls.maxPolarAngle = Math.PI * 0.92;

    let frame = 0;
    const t0 = performance.now();

    function resize() {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (w < 1 || h < 1) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(w, h, false);
    }

    const ro = new ResizeObserver(resize);
    ro.observe(mount);
    resize();

    function tick(now: number) {
      frame = requestAnimationFrame(tick);
      const t = (now - t0) * 0.001;
      knot.rotation.x = t * 0.31;
      knot.rotation.y = t * 0.52;
      wire.rotation.y = -t * 0.11;
      wire.rotation.z = t * 0.07;
      stars.rotation.y = t * 0.018;
      rim.position.x = Math.sin(t * 0.9) * 2.8;
      rim.position.z = Math.cos(t * 0.9) * 2.2 + 0.5;
      controls.update();
      renderer.render(scene, camera);
    }
    frame = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      controls.dispose();
      knotGeo.dispose();
      knotMat.dispose();
      wireGeo.dispose();
      wireMat.dispose();
      starGeo.dispose();
      starMat.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div className="stack-gap">
      <section className="card">
        <div className="cardH">
          <h2>Three.js lab</h2>
          <div className="cardH-meta">Orbit with pointer · ACES tone map · physical + iridescent material</div>
        </div>
        <div className="cardB" style={{ padding: 0 }}>
          <div ref={mountRef} className="three-stage" role="img" aria-label="Interactive 3D torus knot and starfield" />
        </div>
      </section>

      <section className="card">
        <div className="cardH">
          <h2>Where to go next</h2>
          <div className="cardH-meta">Inspiration</div>
        </div>
        <div className="cardB">
          <ul className="how-list">
            <li>
              <a href="https://threejs.org/examples/" target="_blank" rel="noreferrer">
                Official examples gallery
              </a>{" "}
              — hundreds of patterns from post-processing to physics.
            </li>
            <li>
              <a href="https://threejs-journey.com/" target="_blank" rel="noreferrer">
                Three.js Journey
              </a>{" "}
              — structured lessons (paid) with shader and R3F depth.
            </li>
            <li>
              <a href="https://discoverthreejs.com/" target="_blank" rel="noreferrer">
                Discover three.js
              </a>{" "}
              — free book-style intro to the same APIs you see here.
            </li>
            <li>
              <a href="https://github.com/pmndrs/react-three-fiber" target="_blank" rel="noreferrer">
                React Three Fiber
              </a>{" "}
              — if you want this scene expressed as declarative React trees.
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
