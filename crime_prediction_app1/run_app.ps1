# PowerShell script to run the Streamlit app
# This uses the full path to Python to avoid Windows Store stub issues

$pythonPath = "C:\Users\maddi\AppData\Local\Programs\Python\Python311\python.exe"

if (Test-Path $pythonPath) {
    Write-Host "Starting Streamlit app..." -ForegroundColor Green
    & $pythonPath -m streamlit run app.py
} else {
    Write-Host "Python not found at $pythonPath" -ForegroundColor Red
    Write-Host "Please update the pythonPath variable in this script with your Python installation path." -ForegroundColor Yellow
}

