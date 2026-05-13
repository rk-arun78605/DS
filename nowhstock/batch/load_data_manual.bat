@echo off
REM ============================================================
REM LOAD ALL DATA - Python Version (Recommended)
REM ============================================================
REM This batch file runs the Python data loading script.
REM Provides better progress tracking and error handling than pure SQL.
REM 
REM Double-click to run manually.
REM ============================================================

cd /d "d:\Dashboard Code\NO_WH\DS"

echo.
echo ============================================================
echo LOAD ALL DATA - Python Version
echo ============================================================
echo.
echo This will load all CSV data into PostgreSQL tables:
echo   - nowhstock_tbl_new
echo   - sup_shop_grn
echo   - itemdetails
echo   - shopexpiry
echo   - whgrndetails
echo   - sit_data
echo   - sales_2025 (LARGE - may take 5-15 minutes)
echo.
echo WARNING: This will TRUNCATE and reload all tables!
echo.
pause

REM Run Python script
python load_all_data.py

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS - All data loaded successfully!
    echo ============================================================
    echo.
    echo Next steps:
    echo   1. python refresh_materialized_views.py
    echo   2. streamlit run nowhstock_ds.py
    echo.
) else (
    echo.
    echo ============================================================
    echo ERROR - Data load failed! Check logs for details.
    echo ============================================================
    echo.
    echo Log file: data_load_log.txt
    echo.
)

echo Press any key to exit...
pause > nul
