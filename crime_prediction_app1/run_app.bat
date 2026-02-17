@echo off
REM Batch script to run the Streamlit app
REM This uses the full path to Python to avoid Windows Store stub issues

set PYTHON_PATH=C:\Users\maddi\AppData\Local\Programs\Python\Python311\python.exe

if exist "%PYTHON_PATH%" (
    echo Starting Streamlit app...
    "%PYTHON_PATH%" -m streamlit run app.py
) else (
    echo Python not found at %PYTHON_PATH%
    echo Please update the PYTHON_PATH variable in this script with your Python installation path.
    pause
)

