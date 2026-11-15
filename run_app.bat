@echo off
echo ========================================================
echo   Student Performance Analysis System - Streamlit Launcher
echo ========================================================
echo.

set PYTHON_EXE=C:\Users\gayat\AppData\Local\Programs\Python\Python313\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo Starting Streamlit Web Application...
"%PYTHON_EXE%" -m streamlit run app.py --server.port 8501 --server.headless false

pause
