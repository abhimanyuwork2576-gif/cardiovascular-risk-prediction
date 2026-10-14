@echo off
title Launch Cardiovascular Risk Prediction - Public Website Link
cd /d "%~dp0"
echo ======================================================================
echo  Track A: Cardiovascular Risk Visualization & Prediction
echo  Starting Server & Generating Global Public Link...
echo ======================================================================
echo.

python share_public.py
if %errorlevel% neq 0 (
    echo.
    echo An error occurred while launching.
    pause
)
