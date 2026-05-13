@echo off
REM Century Stock Penetration - Daily Update
REM Loads daily sales (gendailysale.csv) and refreshes all views

echo ========================================
echo CENTURY DAILY UPDATE
echo ========================================
echo.

cd /d "d:\Dashboard Code\NO_WH\DS\centurypenetration"

echo Step 1: Loading daily sales data...
echo Step 2: Refreshing materialized views...
echo.

python daily_update.py

echo.
echo ========================================
echo DAILY UPDATE COMPLETE
echo ========================================
echo.

pause
