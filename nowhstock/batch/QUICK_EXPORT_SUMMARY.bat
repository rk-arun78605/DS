@echo off
REM Quick Export - Recommendations Comparison Summary
REM This creates a manageable Excel file with summaries and top items

echo ================================================================================
echo EXPORT RECOMMENDATIONS COMPARISON SUMMARY
echo ================================================================================
echo.
echo This will export:
echo   - Summary statistics (Production vs Test)
echo   - Priority-to-priority transfers (NEW)
echo   - Shop breakdowns
echo   - Top 1000 items from each view
echo   - Top 500 priority-to-priority items
echo.
echo NOTE: Full data (1.3M rows) is too large for Excel.
echo       This exports summaries only for validation.
echo.

cd /d "%~dp0"

python export_recommendations_summary.py

if errorlevel 1 (
    echo.
    echo ERROR: Export failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Opening Excel file...
echo ================================================================================
echo.

REM Open the most recent Excel file
for /f "delims=" %%i in ('dir /b /o-d "Recommendations_Comparison_Summary_*.xlsx" 2^>nul') do (
    start "" "%%i"
    goto :done
)

:done
echo File opened. Review the sheets:
echo   1. Summary - Key metrics comparison
echo   2. Source Shop Comparison - Priority shops highlighted
echo   3. Priority-to-Priority - NEW transfers between priority shops
echo   4. Priority Items (Top 500) - Detailed items
echo.
pause
