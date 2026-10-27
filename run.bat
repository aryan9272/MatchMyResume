@echo off
title MatchMyResume - ATS Resume Matcher
color 0b
echo =====================================================================
echo                      MATCH MY RESUME
echo          AI-Powered ATS Resume Matcher & NLP Analyzer
echo =====================================================================
echo.
echo [1/2] Setting working directory...
cd /d "%~dp0"

echo [2/2] Verifying Python Environment...
if exist "venv\Scripts\python.exe" (
    echo [OK] Found Python Virtual Environment.
    echo.
    echo =====================================================================
    echo   Launching Streamlit Dashboard...
    echo   URL: http://localhost:8501
    echo   (Press Ctrl+C in this terminal window to stop the server)
    echo =====================================================================
    echo.
    venv\Scripts\python.exe -m streamlit run app.py
) else (
    echo [INFO] Local venv not detected. Attempting to run with system Python...
    python -m streamlit run app.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The application stopped unexpectedly with error code: %ERRORLEVEL%
    echo Please make sure dependencies are installed using: pip install -r requirements.txt
    echo.
    pause
)
