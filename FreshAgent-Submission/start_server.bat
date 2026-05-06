@echo off
echo ============================================================
echo   FreshAgent - Fruits ^& Vegetables Adulteration Detection
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [Step 1] Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment. Make sure Python 3.10+ is installed.
        pause & exit /b 1
    )
)

echo [Step 2] Activating virtual environment...
call venv\Scripts\activate.bat

echo [Step 3] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [Step 4] Starting FastAPI server...
echo.
echo  Web Dashboard will be available at:
echo    http://localhost:9090
echo.
echo  API Docs available at:
echo    http://localhost:9090/docs
echo.
echo  Press Ctrl+C to stop the server.
echo ============================================================
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 9090 --reload

pause
