@echo off
REM ============================================================
REM Run Barcode Matcher Application
REM ============================================================

echo ========================================
echo Starting Barcode Matcher App
echo ========================================
echo.
echo Dashboard will open in your browser...
echo URL: http://localhost:8503
echo.
echo Press Ctrl+C to stop the application
echo ========================================
echo.

cd /d "d:\Dashboard Code\NO_WH\DS\barcode_matcher"
streamlit run barcode_matcher_app.py --server.port 8503
