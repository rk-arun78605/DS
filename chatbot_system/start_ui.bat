@echo off
REM Streamlit UI Startup Script

echo.
echo ====================================================
echo     Streamlit UI - Frontend
echo ====================================================
echo.

REM Get Python executable path
set PYTHON_EXE=..\..\.venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo Error: Virtual environment not found!
    echo Please check your Python installation
    pause
    exit /b 1
)

echo.
echo Starting Streamlit Frontend...
echo Opening http://localhost:8501 in browser...
echo.

%PYTHON_EXE% -m streamlit run streamlit_ui.py --logger.level=error

pause
