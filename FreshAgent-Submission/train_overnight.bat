@echo off
title FreshAgent Overnight Training (Auto-Resume Enabled)
echo ================================================
echo   FreshAgent - Overnight Training
echo   Started: %date% %time%
echo ================================================
echo.
echo Settings : 7 epochs, batch_size=16, lr=0.0001
echo Dataset  : Augmented-Resized Image
echo Output   : models/
echo.
echo AUTO-RESUME: If a previous checkpoint exists in
echo   models\resume_checkpoint.pth  it will continue
echo   from exactly where it stopped!
echo.
echo Training will take approx 6-7 hours.
echo DO NOT close this window!
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat

python training\train.py ^
    --data_dir "Augmented-Resized Image" ^
    --output_dir models ^
    --epochs 7 ^
    --batch_size 16 ^
    --lr 0.0001 ^
    --patience 5

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================
    echo   Training COMPLETE! Exporting to ONNX...
    echo ================================================
    python training\export.py --model_path models\best_model.pth --output_path models\freshagent.onnx
    echo.
    echo   Generating plots...
    python training\plot_results.py --output_dir models
    echo.
    echo ================================================
    echo   ALL DONE! Model saved to models\freshagent.onnx
    echo   Finished: %date% %time%
    echo ================================================
) ELSE (
    echo.
    echo ================================================
    echo   Training stopped or errored.
    echo   Resume checkpoint saved - just run this
    echo   file again to continue from last epoch!
    echo ================================================
)

pause
