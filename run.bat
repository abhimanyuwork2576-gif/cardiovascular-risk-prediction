@echo off
title Track A: Cardiovascular Risk Visualization & Prediction
echo ======================================================================
echo  Track A: Cardiovascular Risk Visualization & Prediction Prototype
echo ======================================================================
echo.
cd /d "%~dp0"
echo Checking Python environment...
python -c "import fastapi, uvicorn, xgboost, shap, sklearn, pandas, numpy, joblib" 2>nul
if %errorlevel% neq 0 (
    echo Installing missing requirements...
    pip install -r requirements.txt
)
echo.
echo Launching Application Server...
python run.py
pause
