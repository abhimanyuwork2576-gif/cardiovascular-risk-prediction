# 🫀 Track A: Cardiovascular Risk Visualization & Stenosis Prediction

> **Clinical Safety Notice:**  
> *For educational and decision-support purposes only. Not a substitute for formal diagnostic imaging.*

An end-to-end, clinical-grade web application prototype for non-invasive coronary artery disease (CAD) risk assessment and individualized vessel stenosis prediction.

This prototype integrates **XGBoost machine learning models** trained on the benchmark **Z-Alizadeh Sani** dataset, real-time **TreeSHAP explainability**, and an interactive **3D WebGL human heart viewer (Three.js)** that dynamically color-codes the three primary coronary arteries:
- **LAD**: Left Anterior Descending Artery
- **LCX**: Left Circumflex Artery
- **RCA**: Right Coronary Artery

---

## 🌟 Key Features

1. **Multi-Target Machine Learning Pipeline**:
   - **Overall CAD Status**: Detects presence of significant CAD ($\ge 50\%$ diameter stenosis in any major vessel).
   - **Individual Vessel Stenosis**: Predicts stenosis risk probabilities for **LAD**, **LCX**, and **RCA** independently.
   - **Target Leakage Shielded**: Strictly excludes `Cath`, `LAD`, `LCX`, and `RCA` from the input feature matrix.

2. **Explainable AI (SHAP)**:
   - Real-time **TreeSHAP** attribution computes exact feature impact ($\Delta \text{Log-odds}$) for each patient.
   - Visualized via an interactive **waterfall/bar chart** distinguishing risk drivers (crimson) from protective factors (emerald).
   - Generates an automated **clinical narrative summary** explaining the biological factors behind the prediction.

3. **3D Interactive Anatomical Heart Viewer (Three.js)**:
   - Renders a 3D heart model with anatomically positioned coronary branches.
   - **Dynamic Color-Coding**:
     - 🟢 **Green** (`#10b981`): Low Stenosis Risk ($< 35\%$)
     - 🟡 **Amber** (`#f59e0b`): Moderate / Borderline Risk ($35\% - 65\%$)
     - 🔴 **Red** (`#ef4444`): High Stenosis Risk ($> 65\%$)
   - **Ischemic Warning Pulse**: High-risk vessels exhibit an emissive pulsation effect.
   - **Cardiac Beat Simulation**: Realistic dual-phase systolic contraction cycle with adjustable toggle.
   - **Camera Presets**: One-click orientation to Anterior (LAD), Lateral (LCX), and Inferior (RCA).
   - **Myocardial Opacity Slider**: Enables semi-transparent X-ray viewing of coronary paths.
   - **Interactive Raycasting**: Hover and click on vessels to inspect risk probability and clinical territories.

4. **Clinical Presets**:
   - Preloaded patient archetypes for rapid evaluation:
     - *Normal / Low Risk Patient*
     - *High Risk / Multi-Vessel CAD Candidate*
     - *Anterior Ischemia / Probable LAD Stenosis*
     - *Moderate Risk / Metabolic Dyslipidemia*

---

## 📂 Project Structure

```text
cardio-risk-cad/
├── app.py                         # FastAPI web server and prediction/SHAP API
├── run.py                         # One-click application launcher
├── requirements.txt               # Python package dependencies
├── README.md                      # Documentation & usage manual
├── data/
│   ├── z_alizadeh_sani_extended.csv   # Benchmark UCI dataset (303 patients, 59 attributes)
│   └── ext411/                    # Extracted dataset archive
├── models/
│   ├── train.py                   # Model training and SHAP pipeline
│   ├── cad_model.joblib           # Trained XGBoost model for overall CAD
│   ├── lad_model.joblib           # Trained XGBoost model for LAD stenosis
│   ├── lcx_model.joblib           # Trained XGBoost model for LCX stenosis
│   ├── rca_model.joblib           # Trained XGBoost model for RCA stenosis
│   ├── shap_explainer.joblib      # Pre-fitted TreeSHAP explainer
│   └── metadata.json              # Feature definitions, default reference values & metrics
└── static/
    ├── index.html                 # Clinical dashboard web interface
    ├── css/
    │   └── style.css              # Modern clinical dark UI stylesheet
    ├── js/
    │   ├── app.js                 # Dashboard controller, API integration & Chart.js logic
    │   ├── heartViewer3D.js       # Three.js 3D Heart & coronary shader controller
    │   └── vendor/                # Bundled Three.js and Chart.js libraries (offline ready)
    └── models/
        └── heart.glb              # Open-source 3D Heart GLB model
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- **Python**: Version 3.10, 3.11, 3.12, or 3.13.
- Modern Web Browser (Chrome, Firefox, Edge, Safari) with WebGL support.

### 2. Installation
Open a terminal in the project directory:

```bash
# Navigate to project folder in PowerShell or Command Prompt:
cd "D:\MANDI_VASCULAR"
# (or cd "D:\MANDI_VASCULAR\cardio-risk-cad")

# Install Python dependencies
pip install -r requirements.txt
```

### 3. (Optional) Re-train Models
The repository includes pre-trained models. If you wish to retrain them on the dataset:
```bash
python models/train.py
```

### 4. Run the Application
**Option A — Double-click Windows batch script:**
Double click `run.bat` in `D:\MANDI_VASCULAR`.

**Option B — Run with Python:**
```bash
python run.py
```

**Option C — Run directly with Uvicorn:**
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The application will start at:
👉 **`http://127.0.0.1:8000`**

Interactive OpenAPI / Swagger documentation is available at:
👉 **`http://127.0.0.1:8000/docs`**

---

## 📡 API Reference

### `POST /predict`
Predicts overall CAD risk and individual stenosis probabilities for LAD, LCX, and RCA, alongside local SHAP feature attributions.

**Request Body (JSON)**:
```json
{
  "Age": 68,
  "Sex": "Male",
  "BMI": 29.5,
  "BP": 155,
  "PR": 82,
  "DM": 1,
  "HTN": 1,
  "Current Smoker": 1,
  "FH": 1,
  "Typical Chest Pain": 1,
  "EF-TTE": 40,
  "Region RWMA": 2,
  "St Depression": 1,
  "FBS": 180,
  "TG": 240,
  "LDL": 155,
  "HDL": 32
}
```

**Response (JSON)**:
```json
{
  "overall_cad": {
    "probability": 0.9764,
    "percentage": 97.6,
    "status": "High Probability of Significant CAD",
    "severity": "high",
    "binary_prediction": 1
  },
  "vessels": {
    "LAD": {
      "name": "Left Anterior Descending Artery",
      "percentage": 97.5,
      "status": "Significant Stenosis (>=50% narrowing)",
      "severity": "high",
      "color": "#ef4444"
    },
    "LCX": {
      "name": "Left Circumflex Artery",
      "percentage": 79.1,
      "status": "Significant Stenosis (>=50% narrowing)",
      "severity": "high",
      "color": "#ef4444"
    },
    "RCA": {
      "name": "Right Coronary Artery",
      "percentage": 71.9,
      "status": "Significant Stenosis (>=50% narrowing)",
      "severity": "high",
      "color": "#ef4444"
    }
  },
  "shap_explanation": {
    "base_value": 0.8844,
    "top_features": [ ... ],
    "narrative": "Primary factors driving elevated risk: Typical Anginal Chest Pain, Patient Age, Regional Wall Motion Abnormality..."
  },
  "disclaimer": "For educational and decision-support purposes only. Not a substitute for formal diagnostic imaging."
}
```

### `POST /explain`
Dedicated SHAP attribution calculation for detailed clinical interpretability audits.

### `GET /patients/presets`
Retrieves pre-configured clinical archetypes (Low Risk, High Risk, Isolated LAD, Moderate).

### `GET /model/info`
Returns model validation metrics (ROC-AUC, accuracy, sensitivity, feature counts) and verifies target leakage isolation.

---

## 🛡️ Model Validation & Safeguards

- **Target Leakage Protection**: The feature extraction logic explicitly purges all ground-truth catheterization labels (`Cath`, `LAD`, `LCX`, `RCA`) prior to feature transformations and model inference.
- **Validation Metrics** (Stratified Test Split):
  - **Overall CAD Model**: ROC-AUC: **0.864**, Accuracy: **82.0%**, Sensitivity: **88.4%**
  - **LAD Stenosis Model**: ROC-AUC: **0.824**, Accuracy: **73.8%**, Sensitivity: **77.8%**
  - **LCX Stenosis Model**: ROC-AUC: **0.740**, Accuracy: **68.9%**
  - **RCA Stenosis Model**: ROC-AUC: **0.780**, Accuracy: **68.9%**

---

## ⚖️ Clinical Safety Disclaimer
**For educational and decision-support purposes only. Not a substitute for formal diagnostic imaging.**  
Coronary artery disease diagnosis requires definitive catheter angiography, CT coronary angiography (CCTA), or functional myocardial perfusion imaging interpreted by qualified medical professionals.
