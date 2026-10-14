@echo off
title Push Project to GitHub
cd /d "%~dp0"
echo ======================================================================
echo  Push Cardiovascular Risk Prediction to GitHub
echo ======================================================================
echo.
echo Make sure you have created an empty repository on GitHub first!
echo Go to: https://github.com/new
echo.
set /p REPO_URL="Enter your GitHub repository URL (e.g. https://github.com/abhimanyuwork2576-gif/cardiovascular-risk-prediction.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] No URL entered. Exiting.
    pause
    exit /b 1
)

echo.
echo Setting remote origin to %REPO_URL%...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo.
echo Pushing code to GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ======================================================================
    echo  SUCCESS! Project successfully pushed to GitHub!
    echo  Now go to https://dashboard.render.com and deploy your web service.
    echo ======================================================================
) else (
    echo.
    echo [ERROR] Git push failed. Please check your credentials or repo URL.
)

pause
