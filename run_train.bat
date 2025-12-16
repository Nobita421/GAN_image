@echo off
if not exist "data\celeba_preprocessed" (
    echo ⚠️  Warning: data\celeba_preprocessed directory not found.
    echo Please create it and add images before training.
    echo.
)
python train.py
pause
