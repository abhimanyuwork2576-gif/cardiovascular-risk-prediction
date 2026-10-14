"""
Cardiovascular Risk & Vessel Stenosis ML Model Training Pipeline
Uses Z-Alizadeh Sani extended dataset (303 patients)
Predicts:
  1. CAD (Overall Cath status: CAD vs Normal)
  2. LAD (Left Anterior Descending Stenosis: Stenotic vs Normal)
  3. LCX (Left Circumflex Stenosis: Stenotic vs Normal)
  4. RCA (Right Coronary Artery Stenosis: Stenotic vs Normal)

Target Leakage Prevention:
  Explicitly excludes 'LAD', 'LCX', 'RCA', and 'Cath' from features.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import shap
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'z_alizadeh_sani_extended.csv')
MODEL_DIR = os.path.dirname(__file__)

BINARY_COLS = [
    'Obesity', 'CRF', 'CVA', 'Airway disease', 'Thyroid Disease', 'CHF', 'DLP',
    'Weak Peripheral Pulse', 'Lung rales', 'Systolic Murmur', 'Diastolic Murmur',
    'Dyspnea', 'Atypical', 'Nonanginal', 'Exertional CP', 'LowTH Ang', 'LVH', 'Poor R Progression'
]

CATEGORICAL_MULTI = ['BBB', 'VHD']
TARGET_COLS = ['Cath', 'LAD', 'LCX', 'RCA']

def preprocess_dataframe(df_raw, fit_metadata=None):
    """
    Cleans and encodes raw dataframe into model-ready numerical format.
    Prevents target leakage by removing TARGET_COLS.
    """
    df = df_raw.copy()
    
    # Drop targets if present
    for t in TARGET_COLS:
        if t in df.columns:
            df = df.drop(columns=[t])
            
    # Binary Yes/No encoding
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().apply(lambda v: 1 if v == 'Y' else 0)
            
    # Sex encoding (Male=1, Female=0)
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].astype(str).str.strip().str.lower().apply(lambda v: 1 if v in ['male', 'm'] else 0)
        
    # Multi-class categorical variables
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_MULTI if c in df.columns], drop_first=True)
    
    # If fit_metadata provided, ensure column consistency
    if fit_metadata is not None:
        expected_cols = fit_metadata['feature_columns']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = fit_metadata['defaults'].get(col, 0.0)
        df = df[expected_cols]
        
    return df.astype(float)


def train_models():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} patient records with {len(df.columns)} raw columns.")
    
    # Verify target columns
    for t in TARGET_COLS:
        if t not in df.columns:
            raise ValueError(f"Target column '{t}' missing from dataset!")
            
    # Targets extraction
    y_dict = {
        'CAD': (df['Cath'].astype(str).str.strip().str.lower() == 'cad').astype(int),
        'LAD': (df['LAD'].astype(str).str.strip().str.lower() == 'stenotic').astype(int),
        'LCX': (df['LCX'].astype(str).str.strip().str.lower() == 'stenotic').astype(int),
        'RCA': (df['RCA'].astype(str).str.strip().str.lower() == 'stenotic').astype(int)
    }
    
    # Feature extraction (Strict exclusion of LAD, LCX, RCA, Cath)
    X_raw = df.drop(columns=TARGET_COLS)
    X = preprocess_dataframe(X_raw)
    feature_columns = X.columns.tolist()
    print(f"Total features extracted (excluding targets): {len(feature_columns)}")
    
    # Compute default clinical values (medians for numerical, mode for categorical)
    defaults = {}
    for col in feature_columns:
        defaults[col] = float(X[col].median())
        
    metadata = {
        'feature_columns': feature_columns,
        'defaults': defaults,
        'metrics': {},
        'targets': list(y_dict.keys()),
        'target_descriptions': {
            'CAD': 'Overall Coronary Artery Disease Status (Cath finding)',
            'LAD': 'Left Anterior Descending Coronary Artery Stenosis (>=50%)',
            'LCX': 'Left Circumflex Coronary Artery Stenosis (>=50%)',
            'RCA': 'Right Coronary Artery Stenosis (>=50%)'
        }
    }
    
    models = {}
    
    # Train model for each target
    for target_name, y in y_dict.items():
        print(f"\n--- Training XGBoost Model for {target_name} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # XGBoost classifier configuration
        scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
        clf = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        clf.fit(X_train, y_train)
        
        # Evaluate
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        acc = float(accuracy_score(y_test, preds))
        auc = float(roc_auc_score(y_test, probs))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        
        print(f"[{target_name}] Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f} | F1: {f1:.3f} | Sensitivity: {rec:.3f}")
        
        metadata['metrics'][target_name] = {
            'accuracy': round(acc, 4),
            'roc_auc': round(auc, 4),
            'f1_score': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'sample_count': len(y),
            'positive_rate': round(float(y.mean()), 4)
        }
        
        # Save model
        model_filename = os.path.join(MODEL_DIR, f"{target_name.lower()}_model.joblib")
        joblib.dump(clf, model_filename)
        models[target_name] = clf
        
    # Fit SHAP explainer on CAD model (primary risk model) and save background dataset
    print("\nComputing SHAP explainer for CAD model...")
    cad_model = models['CAD']
    explainer = shap.TreeExplainer(cad_model)
    
    # Compute global feature importance across training set
    shap_values_sample = explainer(X)
    global_importance = np.abs(shap_values_sample.values).mean(axis=0)
    top_feature_indices = np.argsort(global_importance)[::-1]
    
    metadata['global_feature_importance'] = [
        {'feature': feature_columns[i], 'importance': round(float(global_importance[i]), 5)}
        for i in top_feature_indices[:20]
    ]
    metadata['cad_base_value'] = float(explainer.expected_value)
    
    # Save metadata
    meta_path = os.path.join(MODEL_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata and metrics to {meta_path}")
    
    # Save explainer
    explainer_path = os.path.join(MODEL_DIR, 'shap_explainer.joblib')
    joblib.dump(explainer, explainer_path)
    print(f"Saved SHAP explainer to {explainer_path}")
    
    print("\nAll models trained and exported successfully!")

if __name__ == '__main__':
    train_models()
