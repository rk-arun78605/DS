@echo off
REM =====================================================
REM AI Analytics Chatbot System - Complete Startup
REM =====================================================
REM This script starts both FastAPI backend and Streamlit frontend

color 0A
cls

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║   🤖 AI Analytics Chatbot System - Startup                 ║
echo ║   Backend: FastAPI (port 8000)                             ║
echo ║   Frontend: Streamlit (port 8501)                          ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if .env exists
if not exist "chatbot_system\.env" (
    echo ❌ ERROR: .env file not found!
    echo.
    echo Please create chatbot_system\.env from chatbot_system\.env.example
    echo.
    echo Steps:
    echo   1. Copy chatbot_system\.env.example to chatbot_system\.env
    echo   2. Update with your OpenAI/Gemini API keys
    echo   3. Run this script again
    echo.
    pause
    exit /b 1
)

echo ✅ Configuration found (.env)
echo.

REM Start FastAPI Backend
echo [1/2] Starting FastAPI Backend on port 8000...
echo       📍 http://localhost:8000
echo       📊 API Docs: http://localhost:8000/docs
echo.

start "FastAPI Backend" cmd /k python chatbot_system/api.py

REM Wait for API to start
echo Waiting for API to initialize...
timeout /t 5 /nobreak

REM Start Streamlit Frontend
echo.
echo [2/2] Starting Streamlit Frontend on port 8501...
echo       🎨 UI: http://localhost:8501
echo.

start "Streamlit UI" cmd /k streamlit run chatbot_system/streamlit_ui.py --logger.level=error

timeout /t 3 /nobreak

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║   ✅ Both services started successfully!                   ║
echo ║                                                             ║
echo ║   🌐 Open in browser:                                       ║
echo ║   → http://localhost:8501                                   ║
echo ║                                                             ║
echo ║   📊 API Documentation:                                     ║
echo ║   → http://localhost:8000/docs                              ║
echo ║                                                             ║
echo ║   Press CTRL+C in any window to stop services             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Keep this window open
pause
