@echo off
REM Century Stock Penetration - Refresh Materialized Views
REM Run this after data updates

echo ========================================
echo REFRESHING CENTURY PENETRATION VIEWS
echo ========================================
echo.

cd /d "d:\Dashboard Code\NO_WH\DS\centurypenetration"

python refresh_century_views.py

echo.
echo ========================================
echo VIEW REFRESH COMPLETE
echo ========================================
echo.

pause
