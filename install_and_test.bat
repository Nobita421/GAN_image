@echo off
echo [1/2] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to install dependencies.
    echo Please ensure you have Python installed and internet access.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/2] Running smoke tests...
python smoke_test.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Smoke tests failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ✅ Setup complete! You can now run 'python train.py'
pause
