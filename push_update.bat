@echo off
echo [1/3] Adding files...
git add .

echo.
echo [2/3] Committing changes...
git commit -m "Fix Keras warning and add dummy data script for Colab"

echo.
echo [3/3] Pushing to GitHub...
git push origin main

echo.
echo ✅ Changes pushed to GitHub.
echo In Colab, run: git pull
pause
