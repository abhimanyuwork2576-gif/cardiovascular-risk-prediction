/**
 * 3D Interactive Cardiovascular Viewer (Three.js)
 * Visualizes 3D Human Heart model (.glb) with dynamic color-coding
 * of the 3 major coronary arteries (LAD, LCX, RCA).
 * 
 * Features:
 * - GLTF/GLB loader with procedural anatomical coronary vessel fallback
 * - Dynamic color-coding based on risk probabilities (Green / Yellow / Red)
 * - Ischemic warning pulse shader/glow on high-risk vessels
 * - Anatomical territory tooltips via Raycaster interaction
 * - Anatomical camera preset angles (Anterior/LAD, Lateral/LCX, Inferior/RCA)
 * - Cardiac heartbeat cycle simulation with BPM control
 * - Myocardial opacity / x-ray transparency toggle
 */

class HeartViewer3D {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container #${containerId} not found!`);
            return;
        }

        this.options = Object.assign({
            modelPath: '/static/models/heart.glb?v=2',
            beatEnabled: true,
            bpm: 72,
            myocardiumOpacity: 0.88
        }, options);

        this.vesselData = {
            LAD: { prob: 0.05, color: 0x10b981, status: 'Normal', name: 'Left Anterior Descending' },
            LCX: { prob: 0.05, color: 0x10b981, status: 'Normal', name: 'Left Circumflex' },
            RCA: { prob: 0.05, color: 0x10b981, status: 'Normal', name: 'Right Coronary Artery' }
        };

        this.meshes = {
            Heart_Body: null,
            LAD: null,
            LCX: null,
            RCA: null
        };

        this.vesselMaterials = {};
        this.heartBodyMaterial = null;
        this.clock = new THREE.Clock();
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.selectedVessel = null;
        this.hoveredVessel = null;

        this.initScene();
        this.setupLights();
        this.loadModel();
        this.setupEventListeners();
        this.animate();
    }

    initScene() {
        const width = this.container.clientWidth || 500;
        const height = this.container.clientHeight || 450;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0f1d); // Deep clinical dark theme

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        this.camera.position.set(0, 0, 9);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.1;
        this.renderer.outputEncoding = THREE.sRGBEncoding;

        this.container.innerHTML = '';
        this.container.appendChild(this.renderer.domElement);

        // OrbitControls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.06;
        this.controls.minDistance = 3.5;
        this.controls.maxDistance = 16.0;
        this.controls.autoRotate = false;
        this.controls.autoRotateSpeed = 1.0;

        // Model root group
        this.heartRoot = new THREE.Group();
        this.scene.add(this.heartRoot);
    }

    setupLights() {
        // Ambient light for soft baseline
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
        this.scene.add(ambientLight);

        // Key light (soft surgical white)
        const keyLight = new THREE.DirectionalLight(0xfff8f2, 1.35);
        keyLight.position.set(5, 7, 8);
        this.scene.add(keyLight);

        // Fill light (clinical cyan tone)
        const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.75);
        fillLight.position.set(-6, -1, -4);
        this.scene.add(fillLight);

        // Rim light (warm epicardial edge definition)
        const rimLight = new THREE.DirectionalLight(0xff7088, 1.1);
        rimLight.position.set(0, -6, 5);
        this.scene.add(rimLight);

        // Soft anterior spotlight to highlight coronary sulci and vessels
        const spotLight = new THREE.PointLight(0xffffff, 0.8, 16);
        spotLight.position.set(0, 2, 7);
        this.scene.add(spotLight);
    }

    loadModel() {
        const loader = new THREE.GLTFLoader();
        
        loader.load(
            this.options.modelPath,
            (gltf) => {
                console.log('3D Heart GLB model loaded successfully');
                this.setupLoadedModel(gltf.scene);
            },
            (xhr) => {
                const percent = Math.round((xhr.loaded / (xhr.total || 1)) * 100);
                this.updateLoadingProgress(percent);
            },
            (error) => {
                console.warn('GLB load failed, falling back to anatomical procedural model:', error);
                this.createProceduralHeart();
            }
        );
    }

    setupLoadedModel(scene) {
        // Hide loader indicator if any
        this.hideLoadingProgress();

        // Traverse scene and extract vessels and myocardium
        this.heartBodyMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x8e1a26,           // Deep physiological myocardium crimson
            roughness: 0.38,
            metalness: 0.10,
            transmission: 0.12,        // Biological subsurface light diffusion
            transparent: true,
            opacity: this.options.myocardiumOpacity,
            reflectivity: 0.5,
            clearcoat: 0.45,           // Moist epicardium tissue gloss
            clearcoatRoughness: 0.20,
            side: THREE.DoubleSide
        });

        // Initialize materials for each vessel
        ['LAD', 'LCX', 'RCA'].forEach(v => {
            this.vesselMaterials[v] = new THREE.MeshStandardMaterial({
                color: this.vesselData[v].color,
                roughness: 0.20,
                metalness: 0.35,
                emissive: this.vesselData[v].color,
                emissiveIntensity: 0.45,
                side: THREE.DoubleSide
            });
        });

        scene.traverse((child) => {
            if (child.isMesh) {
                const name = (child.name || '').toUpperCase();
                child.castShadow = true;
                child.receiveShadow = true;

                if (name.includes('LAD')) {
                    child.material = this.vesselMaterials.LAD;
                    child.userData = { vessel: 'LAD' };
                    this.meshes.LAD = child;
                } else if (name.includes('LCX')) {
                    child.material = this.vesselMaterials.LCX;
                    child.userData = { vessel: 'LCX' };
                    this.meshes.LCX = child;
                } else if (name.includes('RCA')) {
                    child.material = this.vesselMaterials.RCA;
                    child.userData = { vessel: 'RCA' };
                    this.meshes.RCA = child;
                } else {
                    child.material = this.heartBodyMaterial;
                    child.userData = { part: 'Heart_Body' };
                    if (!this.meshes.Heart_Body) {
                        this.meshes.Heart_Body = child;
                    }
                }
            }
        });

        this.heartRoot.add(scene);

        // If any vessel meshes were missing in the GLB, add procedural vessels to ensure LAD/LCX/RCA are visible
        if (!this.meshes.LAD || !this.meshes.LCX || !this.meshes.RCA) {
            console.log('Augmenting with procedural coronary artery spline trees...');
            this.addProceduralCoronaryArteries();
        }

        this.applyVesselColors();
    }

    createProceduralHeart() {
        this.hideLoadingProgress();
        console.log('Generating procedural anatomical heart...');

        // Heart body (Left & Right Ventricles + Atria)
        const heartGeo = new THREE.DodecahedronGeometry(2.2, 3);
        // Distort vertices slightly for cardiac ventricle shape
        const pos = heartGeo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i);
            let y = pos.getY(i);
            let z = pos.getZ(i);

            // Apex taper downward
            if (y < 0) {
                const taper = 1.0 + (y / 2.5) * 0.45;
                x *= Math.max(taper, 0.4);
                z *= Math.max(taper, 0.4);
            }
            // Left ventricle bulge
            if (x > 0) x *= 1.15;
            pos.setXYZ(i, x, y, z);
        }
        heartGeo.computeVertexNormals();

        this.heartBodyMaterial = new THREE.MeshPhysicalMaterial({
            color: 0xa82030,
            roughness: 0.35,
            metalness: 0.15,
            transmission: 0.25,
            transparent: true,
            opacity: this.options.myocardiumOpacity,
            reflectivity: 0.5,
            clearcoat: 0.3
        });

        const heartMesh = new THREE.Mesh(heartGeo, this.heartBodyMaterial);
        heartMesh.userData = { part: 'Heart_Body' };
        this.meshes.Heart_Body = heartMesh;
        this.heartRoot.add(heartMesh);

        // Aorta root
        const aortaCurve = new THREE.CubicBezierCurve3(
            new THREE.Vector3(0.1, 1.6, 0.2),
            new THREE.Vector3(0.2, 2.7, 0.1),
            new THREE.Vector3(-0.8, 2.9, -0.4),
            new THREE.Vector3(-1.2, 2.1, -0.8)
        );
        const aortaGeo = new THREE.TubeGeometry(aortaCurve, 24, 0.48, 16, false);
        const aortaMat = new THREE.MeshStandardMaterial({ color: 0xcc3333, roughness: 0.3, metalness: 0.2 });
        const aortaMesh = new THREE.Mesh(aortaGeo, aortaMat);
        this.heartRoot.add(aortaMesh);

        // Pulmonary artery root
        const paCurve = new THREE.CubicBezierCurve3(
            new THREE.Vector3(-0.4, 1.4, 0.6),
            new THREE.Vector3(-0.5, 2.3, 0.5),
            new THREE.Vector3(0.4, 2.5, -0.2),
            new THREE.Vector3(1.1, 2.2, -0.6)
        );
        const paGeo = new THREE.TubeGeometry(paCurve, 24, 0.42, 16, false);
        const paMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.3, metalness: 0.2 });
        const paMesh = new THREE.Mesh(paGeo, paMat);
        this.heartRoot.add(paMesh);

        this.addProceduralCoronaryArteries();
        this.applyVesselColors();
    }

    addProceduralCoronaryArteries() {
        // 1. LAD (Left Anterior Descending)
        // Descends down the anterior interventricular groove towards apex
        const ladCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(0.05, 1.65, 0.95),
            new THREE.Vector3(0.25, 1.15, 1.55),
            new THREE.Vector3(0.40, 0.45, 1.72),
            new THREE.Vector3(0.48, -0.35, 1.60),
            new THREE.Vector3(0.45, -1.15, 1.30),
            new THREE.Vector3(0.35, -1.85, 0.85),
            new THREE.Vector3(0.20, -2.25, 0.35)
        ]);
        const ladGeo = new THREE.TubeGeometry(ladCurve, 40, 0.12, 12, false);
        this.vesselMaterials.LAD = new THREE.MeshStandardMaterial({
            color: this.vesselData.LAD.color,
            roughness: 0.2,
            metalness: 0.35,
            emissive: this.vesselData.LAD.color,
            emissiveIntensity: 0.3
        });
        const ladMesh = new THREE.Mesh(ladGeo, this.vesselMaterials.LAD);
        ladMesh.userData = { vessel: 'LAD' };
        this.meshes.LAD = ladMesh;
        this.heartRoot.add(ladMesh);

        // LAD Diagonal branches
        const diagCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(0.40, 0.45, 1.72),
            new THREE.Vector3(0.85, 0.15, 1.50),
            new THREE.Vector3(1.20, -0.40, 1.10)
        ]);
        const diagGeo = new THREE.TubeGeometry(diagCurve, 16, 0.07, 8, false);
        const diagMesh = new THREE.Mesh(diagGeo, this.vesselMaterials.LAD);
        diagMesh.userData = { vessel: 'LAD' };
        this.heartRoot.add(diagMesh);

        // 2. LCX (Left Circumflex)
        // Curves laterally around the left atrioventricular groove to posterior
        const lcxCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(0.05, 1.65, 0.95),
            new THREE.Vector3(0.85, 1.35, 1.25),
            new THREE.Vector3(1.65, 0.95, 0.70),
            new THREE.Vector3(1.95, 0.45, -0.15),
            new THREE.Vector3(1.80, -0.20, -1.05),
            new THREE.Vector3(1.25, -0.85, -1.45)
        ]);
        const lcxGeo = new THREE.TubeGeometry(lcxCurve, 40, 0.11, 12, false);
        this.vesselMaterials.LCX = new THREE.MeshStandardMaterial({
            color: this.vesselData.LCX.color,
            roughness: 0.2,
            metalness: 0.35,
            emissive: this.vesselData.LCX.color,
            emissiveIntensity: 0.3
        });
        const lcxMesh = new THREE.Mesh(lcxGeo, this.vesselMaterials.LCX);
        lcxMesh.userData = { vessel: 'LCX' };
        this.meshes.LCX = lcxMesh;
        this.heartRoot.add(lcxMesh);

        // LCX Obtuse Marginal branch
        const omCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(1.95, 0.45, -0.15),
            new THREE.Vector3(1.85, -0.35, 0.35),
            new THREE.Vector3(1.50, -1.25, 0.65)
        ]);
        const omGeo = new THREE.TubeGeometry(omCurve, 16, 0.07, 8, false);
        const omMesh = new THREE.Mesh(omGeo, this.vesselMaterials.LCX);
        omMesh.userData = { vessel: 'LCX' };
        this.heartRoot.add(omMesh);

        // 3. RCA (Right Coronary Artery)
        // Descends in the right atrioventricular sulcus to the inferior surface
        const rcaCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-0.15, 1.60, 0.90),
            new THREE.Vector3(-0.95, 1.35, 1.20),
            new THREE.Vector3(-1.60, 0.85, 0.85),
            new THREE.Vector3(-1.75, 0.05, 0.40),
            new THREE.Vector3(-1.60, -0.75, -0.25),
            new THREE.Vector3(-1.10, -1.40, -0.85),
            new THREE.Vector3(-0.40, -1.95, -1.15)
        ]);
        const rcaGeo = new THREE.TubeGeometry(rcaCurve, 40, 0.11, 12, false);
        this.vesselMaterials.RCA = new THREE.MeshStandardMaterial({
            color: this.vesselData.RCA.color,
            roughness: 0.2,
            metalness: 0.35,
            emissive: this.vesselData.RCA.color,
            emissiveIntensity: 0.3
        });
        const rcaMesh = new THREE.Mesh(rcaGeo, this.vesselMaterials.RCA);
        rcaMesh.userData = { vessel: 'RCA' };
        this.meshes.RCA = rcaMesh;
        this.heartRoot.add(rcaMesh);

        // RCA Acute Marginal branch
        const amCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-1.75, 0.05, 0.40),
            new THREE.Vector3(-1.50, -0.65, 0.85),
            new THREE.Vector3(-1.10, -1.35, 1.05)
        ]);
        const amGeo = new THREE.TubeGeometry(amCurve, 16, 0.07, 8, false);
        const amMesh = new THREE.Mesh(amGeo, this.vesselMaterials.RCA);
        amMesh.userData = { vessel: 'RCA' };
        this.heartRoot.add(amMesh);
    }

    updateVesselRisks(vessels) {
        if (!vessels) return;
        ['LAD', 'LCX', 'RCA'].forEach(v => {
            if (vessels[v]) {
                const info = vessels[v];
                this.vesselData[v].prob = info.probability;
                this.vesselData[v].status = info.status;
                this.vesselData[v].percentage = info.percentage;
                this.vesselData[v].color = this.hexStringToNumber(info.color);
            }
        });
        this.applyVesselColors();
    }

    hexStringToNumber(hexStr) {
        if (typeof hexStr === 'number') return hexStr;
        if (!hexStr) return 0x10b981;
        return parseInt(hexStr.replace('#', '0x'), 16);
    }

    applyVesselColors() {
        ['LAD', 'LCX', 'RCA'].forEach(v => {
            const mat = this.vesselMaterials[v];
            if (mat) {
                const col = new THREE.Color(this.vesselData[v].color);
                mat.color.copy(col);
                mat.emissive.copy(col);
                
                // If high risk (>65%), increase baseline emissive glow
                if (this.vesselData[v].prob > 0.65) {
                    mat.emissiveIntensity = 0.65;
                } else if (this.vesselData[v].prob > 0.35) {
                    mat.emissiveIntensity = 0.40;
                } else {
                    mat.emissiveIntensity = 0.20;
                }
            }
        });
    }

    setMyocardiumOpacity(val) {
        this.options.myocardiumOpacity = parseFloat(val);
        if (this.heartBodyMaterial) {
            this.heartBodyMaterial.opacity = this.options.myocardiumOpacity;
        }
    }

    setCameraPreset(preset) {
        const duration = 1000;
        let targetPos = new THREE.Vector3(0, 0, 9);
        let targetLook = new THREE.Vector3(0, 0, 0);

        switch (preset.toLowerCase()) {
            case 'anterior':
            case 'lad':
                // Focus on anterior interventricular sulcus (LAD) and apex
                targetPos.set(0.2, -0.3, 8.5);
                break;
            case 'lateral':
            case 'lcx':
                // Focus on left lateral surface (LCX) and marginal branches
                targetPos.set(7.5, 0.6, 3.8);
                break;
            case 'inferior':
            case 'rca':
                // Focus on right coronary / inferior surface (RCA) and PDA
                targetPos.set(-7.2, -0.8, 4.0);
                break;
            case 'posterior':
                targetPos.set(0.2, 0.2, -8.5);
                break;
            case 'reset':
            default:
                targetPos.set(0.2, 0.0, 8.5);
                break;
        }

        this.smoothCameraTransition(targetPos, targetLook, duration);
    }

    smoothCameraTransition(endPos, endLook, duration) {
        const startPos = this.camera.position.clone();
        const startTime = performance.now();

        const step = (time) => {
            const elapsed = time - startTime;
            const progress = Math.min(elapsed / duration, 1.0);
            // Ease-in-out curve
            const ease = progress < 0.5 
                ? 2 * progress * progress 
                : -1 + (4 - 2 * progress) * progress;

            this.camera.position.lerpVectors(startPos, endPos, ease);
            this.controls.target.lerp(endLook, 0.05);
            this.controls.update();

            if (progress < 1.0) {
                requestAnimationFrame(step);
            }
        };
        requestAnimationFrame(step);
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize());

        // Raycasting for interactive hover & click
        this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.renderer.domElement.addEventListener('click', (e) => this.onMouseClick(e));
    }

    onWindowResize() {
        if (!this.container) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.heartRoot.children, true);

        let hitVessel = null;
        for (let hit of intersects) {
            if (hit.object.userData && hit.object.userData.vessel) {
                hitVessel = hit.object.userData.vessel;
                break;
            }
        }

        if (hitVessel !== this.hoveredVessel) {
            this.hoveredVessel = hitVessel;
            this.renderer.domElement.style.cursor = hitVessel ? 'pointer' : 'default';
            this.updateHoverTooltip(event, hitVessel);
        }
    }

    onMouseClick(event) {
        if (this.hoveredVessel) {
            this.selectedVessel = this.hoveredVessel;
            if (typeof this.options.onVesselSelect === 'function') {
                this.options.onVesselSelect(this.hoveredVessel, this.vesselData[this.hoveredVessel]);
            }
        }
    }

    updateHoverTooltip(event, vesselKey) {
        let tooltip = document.getElementById('vessel-3d-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'vessel-3d-tooltip';
            tooltip.className = 'vessel-tooltip';
            this.container.appendChild(tooltip);
        }

        if (!vesselKey) {
            tooltip.style.display = 'none';
            return;
        }

        const data = this.vesselData[vesselKey];
        const rect = this.container.getBoundingClientRect();
        const x = event.clientX - rect.left + 15;
        const y = event.clientY - rect.top + 15;

        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
        tooltip.style.display = 'block';

        const colorHex = '#' + data.color.toString(16).padStart(6, '0');
        tooltip.innerHTML = `
            <div style="font-weight: 700; color: ${colorHex}; margin-bottom: 2px;">
                ${vesselKey} - ${data.name}
            </div>
            <div style="font-size: 11px; color: #94a3b8;">
                Predicted Stenosis Risk: <strong>${data.percentage || (data.prob * 100).toFixed(1)}%</strong>
            </div>
            <div style="font-size: 11px; color: #cbd5e1; margin-top: 2px;">
                Status: <strong>${data.status}</strong>
            </div>
        `;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();
        const elapsedTime = this.clock.getElapsedTime();

        // 1. Cardiac Contraction (Heartbeat) Simulation
        if (this.options.beatEnabled && this.heartRoot) {
            const beatFreq = (this.options.bpm / 60) * Math.PI * 2;
            // Characteristic double-peak waveform (atrial + ventricular systole)
            const phase = (elapsedTime * beatFreq) % (Math.PI * 2);
            let contraction = 0.0;
            if (phase < Math.PI * 0.4) {
                contraction = Math.sin((phase / (Math.PI * 0.4)) * Math.PI) * 0.045;
            } else if (phase > Math.PI * 0.5 && phase < Math.PI * 0.9) {
                contraction = Math.sin(((phase - Math.PI * 0.5) / (Math.PI * 0.4)) * Math.PI) * 0.025;
            }

            const scaleVal = 1.0 + contraction;
            this.heartRoot.scale.set(scaleVal, 1.0 - contraction * 0.5, scaleVal);
        }

        // 2. High-Risk Ischemia Warning Pulse Shader Effect
        ['LAD', 'LCX', 'RCA'].forEach(v => {
            const data = this.vesselData[v];
            const mat = this.vesselMaterials[v];
            if (mat && data.prob > 0.65) {
                // Pulsate emissive glow between 0.35 and 0.85
                const pulse = 0.55 + 0.30 * Math.sin(elapsedTime * 4.5);
                mat.emissiveIntensity = pulse;
            }
        });

        // 3. Update controls and render
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    updateLoadingProgress(percent) {
        let bar = document.getElementById('model-progress-bar');
        if (bar) bar.style.width = `${percent}%`;
    }

    hideLoadingProgress() {
        const overlay = document.getElementById('model-loading-overlay');
        if (overlay) overlay.style.display = 'none';
    }
}

window.HeartViewer3D = HeartViewer3D;
