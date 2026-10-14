"""
Track A: Cardiovascular Risk Visualization & Prediction
FastAPI Backend with XGBoost, Multi-Vessel Stenosis (LAD, LCX, RCA),
TreeSHAP Feature Attribution, and Clinical Safety Safeguards.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')

app = FastAPI(
    title="Cardiovascular Risk & Vessel Stenosis Prediction API",
    description="Machine Learning CAD and Coronary Artery Stenosis (LAD, LCX, RCA) prediction using Z-Alizadeh Sani dataset with SHAP explanations.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load metadata, models, and SHAP explainer
with open(METADATA_PATH, 'r') as f:
    METADATA = json.load(f)

FEATURE_COLUMNS = METADATA['feature_columns']
DEFAULTS = METADATA['defaults']

MODELS = {
    'cad': joblib.load(os.path.join(MODEL_DIR, 'cad_model.joblib')),
    'lad': joblib.load(os.path.join(MODEL_DIR, 'lad_model.joblib')),
    'lcx': joblib.load(os.path.join(MODEL_DIR, 'lcx_model.joblib')),
    'rca': joblib.load(os.path.join(MODEL_DIR, 'rca_model.joblib'))
}

SHAP_EXPLAINER = joblib.load(os.path.join(MODEL_DIR, 'shap_explainer.joblib'))

CLINICAL_DISCLAIMER = (
    "For educational and decision-support purposes only. "
    "Not a substitute for formal diagnostic imaging."
)

# Friendly readable titles for key clinical features
FEATURE_HUMAN_NAMES = {
    'Typical Chest Pain': 'Typical Anginal Chest Pain',
    'Age': 'Patient Age (years)',
    'EF-TTE': 'Ejection Fraction (EF %)',
    'ESR': 'Erythrocyte Sedimentation Rate (ESR)',
    'FBS': 'Fasting Blood Sugar (mg/dL)',
    'Region RWMA': 'Regional Wall Motion Abnormality',
    'TG': 'Serum Triglycerides (mg/dL)',
    'HTN': 'Hypertension History',
    'Tinversion': 'T-Wave Inversion on ECG',
    'BP': 'Systolic Blood Pressure (mmHg)',
    'LDL': 'LDL Cholesterol (mg/dL)',
    'HDL': 'HDL Cholesterol (mg/dL)',
    'DM': 'Diabetes Mellitus',
    'Sex': 'Biological Sex (1=Male, 0=Female)',
    'BMI': 'Body Mass Index (BMI)',
    'PR': 'Resting Pulse Rate (bpm)',
    'St Depression': 'ST-Segment Depression',
    'St Elevation': 'ST-Segment Elevation',
    'Q Wave': 'Pathological Q Waves',
    'Current Smoker': 'Active Smoking',
    'FH': 'Family History of CAD',
    'CR': 'Serum Creatinine (mg/dL)',
    'BUN': 'Blood Urea Nitrogen (mg/dL)',
    'HB': 'Hemoglobin Level (g/dL)',
    'WBC': 'White Blood Cell Count',
    'PLT': 'Platelet Count',
    'Function Class': 'NYHA Functional Class (0-3)',
    'Dyspnea': 'Dyspnea / Shortness of Breath'
}

def get_vessel_color_and_severity(prob: float) -> Dict[str, Any]:
    """
    Categorizes risk probability into clinical severity and color code.
    Green: Low Risk (< 0.35)
    Amber/Yellow: Moderate Risk (0.35 - 0.65)
    Red: High Risk (> 0.65)
    """
    if prob < 0.35:
        return {
            'color': '#10b981',       # Emerald Green
            'hex': 0x10b981,
            'category': 'Low Risk',
            'severity': 'low',
            'stenosis_status': 'Likely Normal (<50% narrowing)'
        }
    elif prob <= 0.65:
        return {
            'color': '#f59e0b',       # Amber / Yellow
            'hex': 0xf59e0b,
            'category': 'Moderate Risk',
            'severity': 'moderate',
            'stenosis_status': 'Equivocal / Borderline Stenosis'
        }
    else:
        return {
            'color': '#ef4444',       # Red
            'hex': 0xef4444,
            'category': 'High Risk',
            'severity': 'high',
            'stenosis_status': 'Significant Stenosis (>=50% narrowing)'
        }

def prepare_feature_vector(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Takes arbitrary clinical patient input, sanitizes it, fills defaults,
    and returns a 1-row DataFrame aligned with FEATURE_COLUMNS.
    Strictly ensures LAD, LCX, RCA, Cath are not included in features.
    """
    row_dict = dict(DEFAULTS)
    
    for key, value in patient_data.items():
        if key in ['Cath', 'LAD', 'LCX', 'RCA']:
            continue # Target leakage defense
            
        # Boolean / string standardizations
        if key == 'Sex':
            if isinstance(value, str):
                row_dict['Sex'] = 1.0 if value.strip().lower() in ['male', 'm', '1'] else 0.0
            else:
                row_dict['Sex'] = float(value)
        elif key in ['BBB', 'VHD']:
            val_str = str(value).strip()
            # Handle one-hot encoding columns
            for col in FEATURE_COLUMNS:
                if col.startswith(f"{key}_"):
                    suffix = col.replace(f"{key}_", "")
                    row_dict[col] = 1.0 if val_str.lower() == suffix.lower() else 0.0
        else:
            if isinstance(value, (bool, np.bool_)):
                val_num = 1.0 if value else 0.0
            elif isinstance(value, str):
                val_clean = value.strip().upper()
                if val_clean in ['Y', 'YES', 'TRUE', '1']:
                    val_num = 1.0
                elif val_clean in ['N', 'NO', 'FALSE', '0']:
                    val_num = 0.0
                else:
                    try:
                        val_num = float(value)
                    except ValueError:
                        val_num = DEFAULTS.get(key, 0.0)
            elif value is not None:
                val_num = float(value)
            else:
                val_num = DEFAULTS.get(key, 0.0)
                
            if key in row_dict:
                row_dict[key] = val_num
                
    # Build dataframe in exact column order
    df_row = pd.DataFrame([row_dict])[FEATURE_COLUMNS].astype(float)
    return df_row


class PatientInput(BaseModel):
    # Core Vitals & Demographics
    Age: Optional[float] = Field(default=55.0, description="Patient age in years (30 - 86)")
    Sex: Optional[Any] = Field(default="Male", description="'Male' or 'Female' / 1 or 0")
    BMI: Optional[float] = Field(default=26.5, description="Body Mass Index (kg/m^2)")
    BP: Optional[float] = Field(default=130.0, description="Systolic Blood Pressure (mmHg)")
    PR: Optional[float] = Field(default=75.0, description="Resting Pulse Rate (bpm)")
    
    # Clinical History
    DM: Optional[Any] = Field(default=0, description="Diabetes Mellitus (1/0 or Y/N)")
    HTN: Optional[Any] = Field(default=1, description="Hypertension History (1/0 or Y/N)")
    Current_Smoker: Optional[Any] = Field(default=0, alias="Current Smoker", description="Active Smoker (1/0 or Y/N)")
    FH: Optional[Any] = Field(default=0, description="Family History of CAD (1/0 or Y/N)")
    DLP: Optional[Any] = Field(default=0, description="Dyslipidemia (1/0 or Y/N)")
    Obesity: Optional[Any] = Field(default=0, description="Obesity (1/0 or Y/N)")
    CRF: Optional[Any] = Field(default=0, description="Chronic Renal Failure (1/0 or Y/N)")
    
    # Symptoms
    Typical_Chest_Pain: Optional[Any] = Field(default=1, alias="Typical Chest Pain", description="Typical Anginal Chest Pain (1/0 or Y/N)")
    Dyspnea: Optional[Any] = Field(default=0, description="Dyspnea / Shortness of breath (1/0 or Y/N)")
    Function_Class: Optional[float] = Field(default=1.0, alias="Function Class", description="NYHA Functional Class (0 - 3)")
    Exertional_CP: Optional[Any] = Field(default=0, alias="Exertional CP", description="Exertional Chest Pain (1/0 or Y/N)")
    Atypical: Optional[Any] = Field(default=0, description="Atypical Chest Pain (1/0 or Y/N)")
    Nonanginal: Optional[Any] = Field(default=0, description="Non-anginal Chest Pain (1/0 or Y/N)")
    
    # ECG & Echo Markers
    EF_TTE: Optional[float] = Field(default=50.0, alias="EF-TTE", description="Left Ventricular Ejection Fraction % (15 - 65)")
    Region_RWMA: Optional[float] = Field(default=0.0, alias="Region RWMA", description="Regional Wall Motion Abnormality score (0 - 4)")
    St_Depression: Optional[Any] = Field(default=0, alias="St Depression", description="ST Depression on ECG (1/0 or Y/N)")
    St_Elevation: Optional[Any] = Field(default=0, alias="St Elevation", description="ST Elevation on ECG (1/0 or Y/N)")
    Tinversion: Optional[Any] = Field(default=0, description="T-wave Inversion on ECG (1/0 or Y/N)")
    Q_Wave: Optional[Any] = Field(default=0, alias="Q Wave", description="Pathological Q Wave (1/0 or Y/N)")
    BBB: Optional[str] = Field(default="N", description="Bundle Branch Block ('N', 'LBBB', 'RBBB')")
    VHD: Optional[str] = Field(default="N", description="Valvular Heart Disease ('N', 'mild', 'Moderate', 'Severe')")
    
    # Laboratory Chemistry
    FBS: Optional[float] = Field(default=110.0, description="Fasting Blood Sugar (mg/dL)")
    TG: Optional[float] = Field(default=160.0, description="Serum Triglycerides (mg/dL)")
    LDL: Optional[float] = Field(default=115.0, description="LDL Cholesterol (mg/dL)")
    HDL: Optional[float] = Field(default=42.0, description="HDL Cholesterol (mg/dL)")
    CR: Optional[float] = Field(default=1.0, description="Serum Creatinine (mg/dL)")
    BUN: Optional[float] = Field(default=16.0, description="Blood Urea Nitrogen (mg/dL)")
    ESR: Optional[float] = Field(default=18.0, description="Erythrocyte Sedimentation Rate (mm/hr)")
    HB: Optional[float] = Field(default=13.5, description="Hemoglobin (g/dL)")
    K: Optional[float] = Field(default=4.2, description="Potassium (mEq/L)")
    Na: Optional[float] = Field(default=140.0, description="Sodium (mEq/L)")
    WBC: Optional[float] = Field(default=7400.0, description="White Blood Cells (/uL)")
    Lymph: Optional[float] = Field(default=32.0, description="Lymphocytes %")
    Neut: Optional[float] = Field(default=60.0, description="Neutrophils %")
    PLT: Optional[float] = Field(default=220.0, description="Platelets (10^3/uL)")

    class Config:
        populate_by_name = True
        extra = "allow"


@app.get("/model/info")
def get_model_info():
    """Returns trained model performance metrics, dataset characteristics, and target safeguards."""
    return {
        "dataset": "Z-Alizadeh Sani Extended Dataset (303 patient records, 59 raw attributes)",
        "safeguards": {
            "target_leakage_prevented": True,
            "excluded_features": ["Cath", "LAD", "LCX", "RCA"],
            "input_feature_count": len(FEATURE_COLUMNS)
        },
        "metrics": METADATA['metrics'],
        "top_global_features": METADATA['global_feature_importance'][:15],
        "disclaimer": CLINICAL_DISCLAIMER
    }


@app.get("/patients/presets")
def get_patient_presets():
    """Returns pre-configured clinical personas for quick clinical evaluation testing."""
    return [
        {
            "id": "healthy_low_risk",
            "name": "Normal / Low Risk Patient",
            "description": "Young asymptomatic patient with normal lipids, blood pressure, and ECG.",
            "data": {
                "Age": 38,
                "Sex": "Female",
                "BMI": 22.4,
                "BP": 115,
                "PR": 68,
                "DM": 0,
                "HTN": 0,
                "Current Smoker": 0,
                "FH": 0,
                "Typical Chest Pain": 0,
                "Dyspnea": 0,
                "Function Class": 0,
                "EF-TTE": 60,
                "Region RWMA": 0,
                "St Depression": 0,
                "St Elevation": 0,
                "Tinversion": 0,
                "Q Wave": 0,
                "FBS": 88,
                "TG": 105,
                "LDL": 82,
                "HDL": 58,
                "ESR": 8,
                "CR": 0.8
            }
        },
        {
            "id": "high_risk_triple_vessel",
            "name": "High Risk / Multi-Vessel CAD Candidate",
            "description": "Elderly diabetic smoker with severe hypertension, typical angina, ST depression, and low EF.",
            "data": {
                "Age": 71,
                "Sex": "Male",
                "BMI": 30.5,
                "BP": 165,
                "PR": 88,
                "DM": 1,
                "HTN": 1,
                "Current Smoker": 1,
                "FH": 1,
                "Typical Chest Pain": 1,
                "Dyspnea": 1,
                "Function Class": 2,
                "EF-TTE": 35,
                "Region RWMA": 3,
                "St Depression": 1,
                "St Elevation": 0,
                "Tinversion": 1,
                "Q Wave": 1,
                "FBS": 210,
                "TG": 340,
                "LDL": 175,
                "HDL": 29,
                "ESR": 42,
                "CR": 1.7
            }
        },
        {
            "id": "isolated_lad_stenosis",
            "name": "Anterior Ischemia / Probable LAD Stenosis",
            "description": "Middle-aged patient with anterior chest angina, anterior ECG changes, and regional hypokinesis.",
            "data": {
                "Age": 58,
                "Sex": "Male",
                "BMI": 27.2,
                "BP": 142,
                "PR": 76,
                "DM": 1,
                "HTN": 1,
                "Current Smoker": 0,
                "FH": 1,
                "Typical Chest Pain": 1,
                "Dyspnea": 0,
                "Function Class": 1,
                "EF-TTE": 45,
                "Region RWMA": 2,
                "St Depression": 0,
                "St Elevation": 1,
                "Tinversion": 1,
                "Q Wave": 1,
                "FBS": 138,
                "TG": 210,
                "LDL": 140,
                "HDL": 36,
                "ESR": 24,
                "CR": 1.1
            }
        },
        {
            "id": "moderate_metabolic",
            "name": "Moderate Risk / Metabolic Dyslipidemia",
            "description": "Hypertensive patient with high triglycerides, low HDL, and borderline exertional chest tightness.",
            "data": {
                "Age": 54,
                "Sex": "Female",
                "BMI": 29.0,
                "BP": 140,
                "PR": 78,
                "DM": 0,
                "HTN": 1,
                "Current Smoker": 0,
                "FH": 1,
                "Typical Chest Pain": 0,
                "Dyspnea": 1,
                "Function Class": 1,
                "EF-TTE": 50,
                "Region RWMA": 1,
                "St Depression": 0,
                "St Elevation": 0,
                "Tinversion": 0,
                "Q Wave": 0,
                "FBS": 118,
                "TG": 260,
                "LDL": 132,
                "HDL": 38,
                "ESR": 18,
                "CR": 1.0
            }
        }
    ]


@app.post("/predict")
def predict_cardio_risk(patient: Dict[str, Any]):
    """
    Predicts overall CAD status and individual coronary artery stenosis (LAD, LCX, RCA).
    Includes SHAP local feature importance for patient prediction explanation.
    """
    try:
        df_row = prepare_feature_vector(patient)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature inputs: {str(e)}")

    # 1. Model Inferences
    cad_prob = float(MODELS['cad'].predict_proba(df_row)[0, 1])
    lad_prob = float(MODELS['lad'].predict_proba(df_row)[0, 1])
    lcx_prob = float(MODELS['lcx'].predict_proba(df_row)[0, 1])
    rca_prob = float(MODELS['rca'].predict_proba(df_row)[0, 1])
    
    # 2. Vessel Categorization & Styling
    lad_meta = get_vessel_color_and_severity(lad_prob)
    lcx_meta = get_vessel_color_and_severity(lcx_prob)
    rca_meta = get_vessel_color_and_severity(rca_prob)
    
    # 3. Overall CAD Assessment
    if cad_prob < 0.35:
        overall_status = "Low Probability of CAD"
        overall_severity = "low"
    elif cad_prob <= 0.65:
        overall_status = "Intermediate / Indeterminate Risk"
        overall_severity = "moderate"
    else:
        overall_status = "High Probability of Significant CAD"
        overall_severity = "high"
        
    # 4. SHAP Local Attribution
    shap_explanation = compute_shap_breakdown(df_row)
    
    return {
        "overall_cad": {
            "probability": round(cad_prob, 4),
            "percentage": round(cad_prob * 100, 1),
            "status": overall_status,
            "severity": overall_severity,
            "binary_prediction": int(cad_prob >= 0.50)
        },
        "vessels": {
            "LAD": {
                "name": "Left Anterior Descending Artery",
                "abbreviation": "LAD",
                "anatomical_territory": "Anterior wall, interventricular septum, and apex",
                "probability": round(lad_prob, 4),
                "percentage": round(lad_prob * 100, 1),
                "status": lad_meta['stenosis_status'],
                "severity": lad_meta['severity'],
                "color": lad_meta['color'],
                "hex": lad_meta['hex']
            },
            "LCX": {
                "name": "Left Circumflex Artery",
                "abbreviation": "LCX",
                "anatomical_territory": "Lateral and posterolateral left ventricle",
                "probability": round(lcx_prob, 4),
                "percentage": round(lcx_prob * 100, 1),
                "status": lcx_meta['stenosis_status'],
                "severity": lcx_meta['severity'],
                "color": lcx_meta['color'],
                "hex": lcx_meta['hex']
            },
            "RCA": {
                "name": "Right Coronary Artery",
                "abbreviation": "RCA",
                "anatomical_territory": "Inferior left ventricle, right ventricle, sinus node",
                "probability": round(rca_prob, 4),
                "percentage": round(rca_prob * 100, 1),
                "status": rca_meta['stenosis_status'],
                "severity": rca_meta['severity'],
                "color": rca_meta['color'],
                "hex": rca_meta['hex']
            }
        },
        "shap_explanation": shap_explanation,
        "disclaimer": CLINICAL_DISCLAIMER
    }


def compute_shap_breakdown(df_row: pd.DataFrame) -> Dict[str, Any]:
    """Computes exact SHAP attributions for the CAD XGBoost model."""
    shap_vals = SHAP_EXPLAINER(df_row)
    values = shap_vals.values[0]
    base_val = float(SHAP_EXPLAINER.expected_value)
    
    feature_attributions = []
    for i, col in enumerate(FEATURE_COLUMNS):
        patient_val = float(df_row.iloc[0, i])
        impact = float(values[i])
        friendly_name = FEATURE_HUMAN_NAMES.get(col, col)
        
        feature_attributions.append({
            "feature": col,
            "displayName": friendly_name,
            "patientValue": patient_val,
            "shapValue": round(impact, 4),
            "absShap": abs(impact),
            "direction": "increases_risk" if impact > 0 else "decreases_risk"
        })
        
    # Sort by absolute SHAP impact
    feature_attributions.sort(key=lambda x: x['absShap'], reverse=True)
    
    top_risk_drivers = [f for f in feature_attributions if f['shapValue'] > 0][:5]
    top_protective_factors = [f for f in feature_attributions if f['shapValue'] < 0][:5]
    
    # Generate human narrative summary
    narrative = []
    if top_risk_drivers:
        driver_names = [f"{d['displayName']} (val: {d['patientValue']})" for d in top_risk_drivers[:3]]
        narrative.append(f"Primary factors driving elevated risk: {', '.join(driver_names)}.")
    if top_protective_factors:
        protective_names = [f"{p['displayName']} (val: {p['patientValue']})" for p in top_protective_factors[:3]]
        narrative.append(f"Protective / favorable factors mitigating risk: {', '.join(protective_names)}.")
        
    return {
        "base_value": round(base_val, 4),
        "top_features": feature_attributions[:12],
        "top_risk_drivers": top_risk_drivers,
        "top_protective_factors": top_protective_factors,
        "narrative": " ".join(narrative)
    }


@app.post("/explain")
def explain_patient(patient: Dict[str, Any]):
    """Dedicated endpoint to calculate SHAP feature importance for a given patient."""
    try:
        df_row = prepare_feature_vector(patient)
        explanation = compute_shap_breakdown(df_row)
        return {
            "status": "success",
            "explanation": explanation,
            "disclaimer": CLINICAL_DISCLAIMER
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Serve static files & index.html
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

