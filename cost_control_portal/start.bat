@echo off
title Melcom Cost Control Portal
color 0A
cd /d "%~dp0"

echo.
echo  Starting Melcom Cost Control Portal...
echo.

:: ── Python: use local .venv if it exists, else Administrator's venv ──────────
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
    goto :run
)
if exist "C:\Users\Administrator\.venv\Scripts\python.exe" (
    set "PY=C:\Users\Administrator\.venv\Scripts\python.exe"
    goto :run
)

:: Fallback: create a local .venv
set "PYEXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PYEXE%" (
    echo  ERROR: Python not found at %PYEXE%
    pause & exit /b 1
)
echo [Setup] Creating local .venv...
"%PYEXE%" -m venv .venv
echo [Setup] Installing packages from C:\pkg_transfer ...
.venv\Scripts\pip install --no-index --find-links "C:\pkg_transfer" django psycopg2-binary pillow imagehash openpyxl --quiet
set "PY=.venv\Scripts\python.exe"

:: ── Anthropic API Key (Claude AI Vision) ─────────────────────────────────
set "ANTHROPIC_API_KEY=sk-ant-api03-YOUR-NEW-KEY-HERE"

:run
echo [1/3] Running migrations...
"%PY%" manage.py migrate --noinput
if errorlevel 1 ( echo  ERROR: Migration failed. & pause & exit /b 1 )

echo [2/3] Setting up initial data...
"%PY%" manage.py setup_costcontrol

echo [3/3] Collecting static files...
"%PY%" manage.py collectstatic --noinput --clear >nul 2>&1

echo.
echo  ┌──────────────────────────────────────────────────┐
echo  │  Running at http://192.168.0.60:8507             │
echo  └──────────────────────────────────────────────────┘
echo.
start "" "http://localhost:8507"
"%PY%" manage.py runserver 0.0.0.0:8507
pause
