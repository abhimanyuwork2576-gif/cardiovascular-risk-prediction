@echo off
title Share Public HTTPS Link - Track A Cardiovascular Risk
cd /d "%~dp0"
echo ======================================================================
echo  Track A: Cardiovascular Risk Visualization & Prediction
echo  Generating Free Public HTTPS Tunnel Link for Remote Sharing
echo ======================================================================
echo.
echo NOTE: Keep this window open while sharing!
echo Make sure your app server is running on http://127.0.0.1:8000 (via run.bat).
echo.
echo Connecting to Cloudflare Global Edge Network...
echo.
echo Look for the line:
echo   Your quick Tunnel has been created! URL: https://xxxx.trycloudflare.com
echo.
echo ----------------------------------------------------------------------
cloudflared.exe tunnel --protocol http2 --url http://127.0.0.1:8000
pause
