@echo off
echo ============================================================
echo   FreshAgent - Model Training Script
echo ============================================================
echo.

REM Activate virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run start_server.bat first to set up.
    pause & exit /b 1
)
call venv\Scripts\activate.bat

REM Default dataset path (the unzipped folder inside FreshAgent)
set DEFAULT_DATASET=%~dp0Augmented-Resized Image
set DATASET_PATH=%1
if "%DATASET_PATH%"=="" (
    echo.
    echo No path given. Using default: "%DEFAULT_DATASET%"
    set DATASET_PATH=%DEFAULT_DATASET%
)

echo.
echo [Training] Dataset: %DATASET_PATH%
echo [Training] Output:  .\models\
echo [Training] This may take 30-120 minutes on Intel Iris Xe.
echo.

python training\train.py ^
  --data_dir "%DATASET_PATH%" ^
  --output_dir ".\models" ^
  --epochs 60 ^
  --batch_size 16 ^
  --lr 0.0001 ^
  --patience 15

if errorlevel 1 (
    echo.
    echo Training failed. Check error above.
) else (
    echo.
    echo ============================================================
    echo [Export] Converting model to ONNX for fast inference...
    python training\export.py ^
      --checkpoint ".\models\best_model.pth" ^
      --output ".\models\freshagent.onnx"

    echo.
    echo [Plots] Generating training curves and confusion matrices...
    python training\plot_results.py --output_dir ".\models"

    echo.
    echo ============================================================
    echo   Training Complete!
    echo   Model saved to: .\models\best_model.pth
    echo   ONNX export:    .\models\freshagent.onnx
    echo   Plots saved to: .\models\
    echo ============================================================
)
pause
