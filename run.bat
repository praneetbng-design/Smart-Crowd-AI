@echo off
title Smart Crowd AI: Real-Time Crowd Monitoring & Stampede Prevention System
echo =====================================================================
echo  Smart Crowd AI: Real-Time Crowd Monitoring & Stampede Prevention
echo =====================================================================
echo.
echo Starting Streamlit application on local server...
echo.

"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.port 8501 --server.headless false

pause
