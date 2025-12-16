@echo off
echo [1/5] Initializing Git repository...
git init
if %errorlevel% neq 0 goto :error

echo.
echo [2/5] Adding files...
git add .
if %errorlevel% neq 0 goto :error

echo.
echo [3/5] Committing files...
git commit -m "Initial commit: Vanilla GAN implementation"
:: Ignore error if nothing to commit
if %errorlevel% neq 0 echo (Nothing to commit or already committed)

echo.
echo [4/5] Setting up remote 'origin'...
git branch -M main
git remote add origin https://github.com/oneocu/GAN_image.git
:: If remote exists, update it
if %errorlevel% neq 0 (
    echo Remote 'origin' already exists. Updating URL...
    git remote set-url origin https://github.com/oneocu/GAN_image.git
)

echo.
echo [5/5] Pushing to GitHub...
echo Note: You may be asked to sign in to GitHub in a browser or enter credentials.
git push -u origin main
if %errorlevel% neq 0 goto :error

echo.
echo ✅ Successfully uploaded to https://github.com/oneocu/GAN_image.git
pause
exit /b 0

:error
echo.
echo ❌ An error occurred. Please check the output above.
echo Ensure Git is installed and you have access to the repository.
pause
exit /b 1
