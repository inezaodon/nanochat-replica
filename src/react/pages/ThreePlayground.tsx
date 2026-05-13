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
    scene.fog = new THREE.FogExp2(0xf4f4f5, 0.028);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 120);
    camera.position.set(0, 0.4, 5.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setClearColor(0xf4f4f5, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);

    const hemi = new THREE.HemisphereLight(0xffffff, 0xe4e4e7, 0.85);
    scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 1.35);
    key.position.set(4, 6, 5);
    key.castShadow = true;
    key.shadow.mapSize.setScalar(1024);
    scene.add(key);

    const rim = new THREE.PointLight(0xea580c, 42, 26, 2);
    rim.position.set(-2.5, 1.2, 2);
    scene.add(rim);

    const knotGeo = new THREE.TorusKnotGeometry(1.05, 0.32, 220, 36, 2, 3);
    const knotMat = new THREE.MeshPhysicalMaterial({
      color: 0xea580c,
      metalness: 0.82,
      roughness: 0.18,
      emissive: 0x9a3412,
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
      color: 0xea580c,
      wireframe: true,
      transparent: true,
      opacity: 0.3,
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
      starCol[i * 3] = 0.35 + t * 0.25;
      starCol[i * 3 + 1] = 0.45 + t * 0.2;
      starCol[i * 3 + 2] = 0.55 + t * 0.35;
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
    <div className="stack-gap three-page">
      <section className="card three-teach">
        <div className="cardH">
          <h2>Three.js lab</h2>
          <div className="cardH-meta">WebGL + orbit controls — read the cards, then explore the scene.</div>
        </div>
        <div className="cardB three-teach-grid">
          <div>
            <h3 className="three-teach-subhead">What you are seeing</h3>
            <ul className="how-list mb-0">
              <li>
                <strong>Camera</strong> — a perspective camera; nothing here is “trained”: it is a fixed lens into a
                small synthetic world.
              </li>
              <li>
                <strong>Orbit controls</strong> — you move the camera around a target; the geometry stays in place while
                your viewpoint changes (useful analogy: different “readings” of the same state).
              </li>
              <li>
                <strong>Torus knot (solid mesh)</strong> — a smooth, closed curve embedded in 3D. Think of it as a
                playful stand-in for a <em>latent trajectory</em>: a path in space, not a claim about exact model
                geometry.
              </li>
              <li>
                <strong>Wireframe icosahedron</strong> — shows connectivity and empty space between edges. High-level
                metaphor only: many relationships in a model are sparse or structured, not a literal cage around data.
              </li>
              <li>
                <strong>Particle starfield</strong> — many independent points in space; a loose visual for “many
                positions / many items,” not a literal embedding cloud.
              </li>
            </ul>
          </div>
          <div>
            <h3 className="three-teach-subhead">Controls legend</h3>
            <ul className="how-list mb-0">
              <li>
                <strong>Drag (primary mouse button)</strong> — orbit: rotate the camera around the focal point.
              </li>
              <li>
                <strong>Scroll / trackpad pinch</strong> — zoom in and out between the configured min and max
                distance.
              </li>
              <li>
                <strong>Right-drag (where supported)</strong> — pan (small translations of the view target).
              </li>
              <li>
                <strong>Keyboard</strong> — not wired in this demo; all navigation is pointer-based.
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="cardH">
          <h2>Scene</h2>
          <div className="cardH-meta">ACES tone map · physical + iridescent material · soft shadows</div>
        </div>
        <div className="cardB" style={{ padding: 0 }}>
          <div ref={mountRef} className="three-stage" role="img" aria-label="Interactive 3D torus knot and starfield" />
        </div>
      </section>

      <section className="card three-teach">
        <div className="cardH">
          <h2>Three.js and transforms in ML</h2>
          <div className="cardH-meta">Honest parallels — math is shared, domains differ.</div>
        </div>
        <div className="cardB">
          <ul className="how-list mb-0">
            <li>
              <strong>Sequences as positions</strong> — a list of token vectors is a discrete path through a
              high-dimensional space; transformers update each position based on others, somewhat like local rules on
              a mesh, but in many dimensions and with learned weights.
            </li>
            <li>
              <strong>Attention as relational mixing</strong> — attention scores say how much each position should
              listen to each other position; it is not Euclidean distance in this scene, but the <em>idea</em> of
              pairwise influence is related.
            </li>
            <li>
              <strong>Matrices everywhere</strong> — Three.js uses matrices for rigid transforms; neural nets use
              matrices for linear maps inside layers. Same linear-algebra tools, different objects.
            </li>
            <li>
              <strong>Why 3D then?</strong> — humans parse spatial layouts quickly; this page is for building geometric
              intuition and comfort with cameras, coordinates, and motion — not for claiming the LLM’s latent space is
              three-dimensional.
            </li>
          </ul>
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
