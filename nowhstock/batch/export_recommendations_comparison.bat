@echo off
REM Export Production vs Test Recommendations to Excel
REM Location: d:\Dashboard Code\NO_WH\DS\nowhstock\batch\export_recommendations_comparison.bat

echo ================================================================================
echo EXPORT RECOMMENDATIONS COMPARISON TO EXCEL
echo ================================================================================
echo.

cd /d "%~dp0"

REM Check if test view exists
echo Checking if test view exists...
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT COUNT(*) FROM mv_recommendations_complete_test LIMIT 1;" >nul 2>&1

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Test view does not exist!
    echo ========================================
    echo.
    echo Please run the test view creation first:
    echo    run_test_priority_to_priority.bat
    echo.
    echo Or manually create the test view:
    echo    psql -U postgres -p 3307 -d salesdata -f ..\..\NowhStock_mv_recommendations_complete_TEST.sql
    echo.
    pause
    exit /b 1
)

echo Test view exists, proceeding with export...
echo.

REM Run Python export script
echo Exporting recommendations to Excel...
python export_recommendations_comparison.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Export failed!
    echo ========================================
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Excel file created in current directory.
echo ================================================================================
echo.
echo Open the file and review:
echo   - Summary Comparison sheet for overview
echo   - Priority-to-Priority sheet for new logic
echo   - NEW Recommendations sheet for additions
echo.

pause
