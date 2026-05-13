@echo off
REM Chatbot System Startup Script

echo.
echo ====================================================
echo     AI Analytics Chatbot System - Startup
echo ====================================================
echo.

REM Check if .env exists
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo Please update .env with your API keys
    pause
)

REM Get Python executable path
set PYTHON_EXE=..\..\.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    echo Error: Virtual environment not found!
    echo Please activate: ..\..\venv\Scripts\activate.bat
    pause
    exit /b 1
)

echo.
echo Starting FastAPI Backend Server...
echo.
%PYTHON_EXE% api.py

pause
