@echo off
echo [1/3] Adding files...
git add .

echo.
echo [2/3] Committing changes...
git commit -m "Suppress TF logs and add Colab guide"

echo.
echo [3/3] Pushing to GitHub...
git push origin main

echo.
echo ✅ Changes pushed to GitHub.
echo In Colab, run: git pull
pause
