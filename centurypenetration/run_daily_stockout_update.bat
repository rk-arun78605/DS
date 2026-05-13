@echo off
REM Daily stockout tracking update
REM Schedule this in Windows Task Scheduler to run daily at 1 AM

echo ============================================================
echo CENTURY PENETRATION - DAILY STOCKOUT UPDATE
echo ============================================================
echo.

cd /d "d:\Dashboard Code\NO_WH\DS\centurypenetration"

REM Run the Python script
python daily_stockout_upsert.py

echo.
echo ============================================================
echo Update completed. Check daily_stockout_upsert.log for details
echo ============================================================
pause
