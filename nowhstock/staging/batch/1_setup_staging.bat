@echo off
REM ============================================================
REM STAGING ENVIRONMENT SETUP
REM Location: D:\Dashboard Code\NO_WH\DS\nowhstock\staging\batch\
REM Purpose: Create all staging materialized views and indexes
REM ============================================================

echo ================================================================================
echo STAGING ENVIRONMENT SETUP - Priority-to-Priority Testing
echo ================================================================================
echo.
echo This will create:
echo   - mv_slow_fast_moving_summary_staging
echo   - mv_recommendations_complete_staging (with priority-to-priority transfers)
echo   - All staging indexes
echo.
echo WARNING: This may take 5-10 minutes to complete
echo.
pause

cd /d "%~dp0"

echo.
echo [1/3] Creating staging materialized views...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -f "..\sql\create_staging_views.sql"

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Failed to create staging views!
    echo ========================================
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Staging views created.
echo ================================================================================
echo.
echo [2/3] Verifying staging views...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT COUNT(*) as staging_recommendations FROM mv_recommendations_complete_staging;"

echo.
echo [3/3] Comparing with production...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT 'Production' as environment, COUNT(*) as count FROM mv_recommendations_complete UNION ALL SELECT 'Staging' as environment, COUNT(*) as count FROM mv_recommendations_complete_staging;"

echo.
echo ================================================================================
echo STAGING SETUP COMPLETE!
echo ================================================================================
echo.
echo Next steps:
echo   1. Run the staging dashboard:
echo      streamlit run ..\nowhstock_ds_STAGING.py
echo.
echo   2. Test the priority-to-priority transfers
echo.
echo   3. Compare results with production dashboard
echo.
echo   4. If approved, promote staging to production
echo.
pause
