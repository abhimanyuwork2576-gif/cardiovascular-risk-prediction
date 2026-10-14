/**
 * Track A: Cardiovascular Risk Visualization & Prediction
 * Frontend Application Controller
 * Handles clinical form inputs, API requests (/predict, /explain),
 * Chart.js SHAP visualization, presets loading, and 3D viewer integration.
 */

let heartViewer = null;
let shapChart = null;
let currentPatientData = {};
let availablePresets = [];

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize 3D Heart Viewer
    init3DHeart();

    // 2. Fetch Presets and Model Information
    await loadPresets();
    await loadModelInfo();

    // 3. Setup UI Event Listeners
    setupEventListeners();

    // 4. Run default prediction for initial state
    runPrediction();
});

function init3DHeart() {
    heartViewer = new HeartViewer3D('heart-canvas-container', {
        modelPath: '/static/models/heart.glb',
        beatEnabled: true,
        bpm: 72,
        myocardiumOpacity: 0.88,
        onVesselSelect: (vesselKey, data) => {
            highlightVesselCard(vesselKey);
        }
    });

    // Camera preset buttons
    document.querySelectorAll('[data-camera-preset]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const preset = e.target.getAttribute('data-camera-preset');
            heartViewer.setCameraPreset(preset);
            document.querySelectorAll('[data-camera-preset]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Opacity slider
    const opacitySlider = document.getElementById('opacity-slider');
    if (opacitySlider) {
        opacitySlider.addEventListener('input', (e) => {
            heartViewer.setMyocardiumOpacity(e.target.value);
            document.getElementById('opacity-val').textContent = `${Math.round(e.target.value * 100)}%`;
        });
    }

    // Heartbeat toggle
    const beatToggle = document.getElementById('beat-toggle');
    if (beatToggle) {
        beatToggle.addEventListener('change', (e) => {
            heartViewer.options.beatEnabled = e.target.checked;
        });
    }
}

async function loadPresets() {
    try {
        const res = await fetch('/patients/presets');
        availablePresets = await res.json();
        
        const presetContainer = document.getElementById('presets-list');
        if (!presetContainer) return;

        presetContainer.innerHTML = '';
        availablePresets.forEach((p, idx) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = `preset-badge ${idx === 1 ? 'active' : ''}`;
            btn.innerHTML = `<span class="preset-dot"></span> ${p.name}`;
            btn.title = p.description;
            btn.addEventListener('click', () => applyPreset(p, btn));
            presetContainer.appendChild(btn);
        });

        // Set default preset to High Risk (Preset 1) to showcase multi-vessel CAD
        if (availablePresets.length > 1) {
            applyPreset(availablePresets[1], presetContainer.children[1]);
        }
    } catch (e) {
        console.warn('Could not load presets:', e);
    }
}

function applyPreset(preset, activeBtn) {
    document.querySelectorAll('.preset-badge').forEach(b => b.classList.remove('active'));
    if (activeBtn) activeBtn.classList.add('active');

    const form = document.getElementById('clinical-form');
    if (!form) return;

    // Fill form elements
    Object.entries(preset.data).forEach(([key, val]) => {
        const input = form.elements[key];
        if (input) {
            if (input.type === 'checkbox') {
                input.checked = (val === 1 || val === 'Y' || val === true);
            } else if (input.type === 'radio') {
                const radio = form.querySelector(`input[name="${key}"][value="${val}"]`);
                if (radio) radio.checked = true;
            } else {
                input.value = val;
            }
        }
    });

    // Trigger prediction with new preset
    runPrediction();
}

async function loadModelInfo() {
    try {
        const res = await fetch('/model/info');
        const info = await res.json();
        
        // Display metrics in model info badge
        const metricsEl = document.getElementById('model-metrics-summary');
        if (metricsEl && info.metrics) {
            metricsEl.innerHTML = `
                <span>CAD ROC-AUC: <strong>${(info.metrics.CAD.roc_auc * 100).toFixed(1)}%</strong></span>
                <span class="divider">|</span>
                <span>Accuracy: <strong>${(info.metrics.CAD.accuracy * 100).toFixed(1)}%</strong></span>
                <span class="divider">|</span>
                <span>Features: <strong>${info.safeguards.input_feature_count}</strong></span>
            `;
        }
    } catch (e) {
        console.warn('Could not load model info:', e);
    }
}

function setupEventListeners() {
    const form = document.getElementById('clinical-form');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            runPrediction();
        });

        // Real-time update on input change
        form.addEventListener('change', () => {
            runPrediction();
        });
    }

    // Reset button
    const resetBtn = document.getElementById('reset-form-btn');
    if (resetBtn && availablePresets.length > 0) {
        resetBtn.addEventListener('click', () => {
            applyPreset(availablePresets[0]);
        });
    }
}

function collectFormData() {
    const form = document.getElementById('clinical-form');
    const data = {};

    Array.from(form.elements).forEach(elem => {
        if (!elem.name) return;
        
        if (elem.type === 'checkbox') {
            data[elem.name] = elem.checked ? 1 : 0;
        } else if (elem.type === 'radio') {
            if (elem.checked) data[elem.name] = elem.value;
        } else if (elem.type === 'number' || elem.classList.contains('num-input')) {
            data[elem.name] = parseFloat(elem.value) || 0;
        } else {
            data[elem.name] = elem.value;
        }
    });

    return data;
}

async function runPrediction() {
    const payload = collectFormData();
    currentPatientData = payload;

    showLoadingState(true);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            throw new Error(`Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        updateDashboard(data);
    } catch (err) {
        console.error('Prediction failed:', err);
        showToast('Prediction calculation failed. Please check inputs.', 'error');
    } finally {
        showLoadingState(false);
    }
}

function updateDashboard(data) {
    // 1. Overall CAD Risk Metric
    updateOverallRisk(data.overall_cad);

    // 2. Individual Vessels (LAD, LCX, RCA)
    updateVesselCards(data.vessels);

    // 3. Update 3D Heart Color-Coding
    if (heartViewer) {
        heartViewer.updateVesselRisks(data.vessels);
    }

    // 4. Update SHAP Chart & Clinical Narrative
    updateShapVisualization(data.shap_explanation);
}

function updateOverallRisk(cad) {
    const scoreVal = document.getElementById('overall-cad-score');
    const scoreStatus = document.getElementById('overall-cad-status');
    const scoreBar = document.getElementById('overall-cad-bar');
    const riskBadge = document.getElementById('overall-risk-badge');

    if (scoreVal) scoreVal.textContent = `${cad.percentage}%`;
    if (scoreStatus) scoreStatus.textContent = cad.status;
    if (scoreBar) {
        scoreBar.style.width = `${cad.percentage}%`;
        if (cad.percentage < 35) {
            scoreBar.style.backgroundColor = '#10b981';
        } else if (cad.percentage <= 65) {
            scoreBar.style.backgroundColor = '#f59e0b';
        } else {
            scoreBar.style.backgroundColor = '#ef4444';
        }
    }

    if (riskBadge) {
        riskBadge.className = `risk-pill ${cad.severity}`;
        riskBadge.textContent = cad.severity.toUpperCase() + ' RISK';
    }
}

function updateVesselCards(vessels) {
    ['LAD', 'LCX', 'RCA'].forEach(v => {
        const info = vessels[v];
        if (!info) return;

        const probEl = document.getElementById(`vessel-${v.toLowerCase()}-prob`);
        const statusEl = document.getElementById(`vessel-${v.toLowerCase()}-status`);
        const barEl = document.getElementById(`vessel-${v.toLowerCase()}-bar`);
        const cardEl = document.getElementById(`vessel-card-${v.toLowerCase()}`);

        if (probEl) probEl.textContent = `${info.percentage}%`;
        if (statusEl) statusEl.textContent = info.status;
        if (barEl) {
            barEl.style.width = `${info.percentage}%`;
            barEl.style.backgroundColor = info.color;
        }

        if (cardEl) {
            cardEl.setAttribute('data-severity', info.severity);
            const dot = cardEl.querySelector('.vessel-status-indicator');
            if (dot) dot.style.backgroundColor = info.color;
        }
    });
}

function highlightVesselCard(vesselKey) {
    const cardId = `vessel-card-${vesselKey.toLowerCase()}`;
    const card = document.getElementById(cardId);
    if (card) {
        card.classList.add('highlight-pulse');
        setTimeout(() => card.classList.remove('highlight-pulse'), 1200);
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function updateShapVisualization(shap) {
    if (!shap) return;

    // 1. Narrative summary
    const narrativeEl = document.getElementById('shap-narrative-text');
    if (narrativeEl) {
        narrativeEl.textContent = shap.narrative || 'Model evaluation complete.';
    }

    // 2. Render / Update Chart.js horizontal bar chart
    renderShapChart(shap.top_features);
}

function renderShapChart(topFeatures) {
    const canvas = document.getElementById('shap-chart-canvas');
    if (!canvas) return;

    // Filter to top 10 features
    const features = (topFeatures || []).slice(0, 10).reverse();

    const labels = features.map(f => f.displayName);
    const dataVals = features.map(f => f.shapValue);
    const bgColors = features.map(f => f.shapValue > 0 ? 'rgba(239, 68, 68, 0.85)' : 'rgba(16, 185, 129, 0.85)');
    const borderColors = features.map(f => f.shapValue > 0 ? '#ef4444' : '#10b981');

    if (shapChart) {
        shapChart.data.labels = labels;
        shapChart.data.datasets[0].data = dataVals;
        shapChart.data.datasets[0].backgroundColor = bgColors;
        shapChart.data.datasets[0].borderColor = borderColors;
        shapChart.update();
        return;
    }

    const ctx = canvas.getContext('2d');
    shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Value (Impact on CAD Prediction)',
                data: dataVals,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#0f172a',
                    titleColor: '#38bdf8',
                    bodyColor: '#f1f5f9',
                    borderColor: '#334155',
                    borderWidth: 1,
                    callbacks: {
                        label: (ctx) => {
                            const feat = features[ctx.dataIndex];
                            const direction = feat.shapValue > 0 ? 'Increases Risk' : 'Lowers Risk';
                            return [
                                `Patient Value: ${feat.patientValue}`,
                                `SHAP Contribution: ${feat.shapValue.toFixed(4)} log-odds`,
                                `Effect: ${direction}`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    ticks: { color: '#94a3b8', font: { size: 11 } },
                    title: {
                        display: true,
                        text: '← Lowers Risk | Increases Risk →',
                        color: '#64748b',
                        font: { size: 11, weight: 'bold' }
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e2e8f0', font: { size: 11, weight: '500' } }
                }
            }
        }
    });
}

function showLoadingState(isLoading) {
    const indicator = document.getElementById('predict-loading-indicator');
    if (indicator) {
        indicator.style.opacity = isLoading ? '1' : '0';
    }
}

function showToast(msg, type = 'info') {
    let toast = document.getElementById('app-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'app-toast';
        toast.className = 'toast-alert';
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.className = `toast-alert show ${type}`;
    setTimeout(() => { toast.className = 'toast-alert'; }, 3500);
}
