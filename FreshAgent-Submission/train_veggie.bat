@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  train_veggie.bat – One-click VeggieAgent training script
REM
REM  Usage:
REM    train_veggie.bat "C:\path\to\your\veggie_dataset"
REM
REM  Dataset must be structured as:
REM    veggie_dataset\
REM      Ginger\
REM        Fresh\Fresh\<images>
REM        Rotten\Rotten\<images>
REM        Adulterated\Adulterated\<images>
REM ─────────────────────────────────────────────────────────────────────────────

SET DATA_DIR=%~1

IF "%DATA_DIR%"=="" (
    echo.
    echo  [ERROR] Please provide the path to your veggie dataset as a parameter.
    echo.
    echo  [ERROR] Please provide the path to your veggie dataset as a parameter.
    echo  Usage: train_veggie.bat "C:\path\to\your\veggie_dataset"
    echo.
    exit /b 1
)

echo.
echo  ============================================================
echo   VeggieAgent Training – Starting
echo  ============================================================
echo  Dataset : %DATA_DIR%
echo  Output  : models\
echo.

REM Activate virtual environment
call venv\Scripts\activate
IF ERRORLEVEL 1 (
    echo [ERROR] Could not activate venv. Run start_server.bat first to set up venv.
    exit /b 1
)

REM ─── Phase 1: Train ──────────────────────────────────────────────────────────
echo [1/2] Training VeggieAgent model ...
python veggie_training/train.py ^
    --data_dir "%DATA_DIR%" ^
    --output_dir models ^
    --epochs 4 ^
    --batch_size 16 ^
    --lr 0.0001 ^
    --patience 15

IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Training failed. Check the output above.
    exit /b 1
)

REM ─── Phase 2: Export to ONNX ─────────────────────────────────────────────────
echo.
echo [2/2] Exporting trained model to ONNX ...
python veggie_training/export.py ^
    --checkpoint models/veggieagent_best.pth ^
    --output models/veggieagent.onnx

IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] ONNX export failed.
    exit /b 1
)

echo.
echo  ============================================================
echo   VeggieAgent Training Complete!
echo   Model saved to: models\veggieagent.onnx
echo   Restart the server to load the new model:  start_server.bat
echo  ============================================================
echo.
