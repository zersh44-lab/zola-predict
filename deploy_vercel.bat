@echo off
title Zola Predict - Vercel Deployment Helper
echo ========================================================
echo       ZOLA PREDICT - VERCEL DEPLOYMENT HELPER
echo ========================================================
echo.
echo Step 1: Checking Vercel authentication...
call npx vercel whoami
if %errorlevel% neq 0 (
    echo.
    echo Please log in to your Vercel account when prompted:
    call npx vercel login
)
echo.
echo Step 2: Deploying to Vercel Production...
call npx vercel --prod
echo.
echo ========================================================
echo Deployment process complete!
echo ========================================================
pause
