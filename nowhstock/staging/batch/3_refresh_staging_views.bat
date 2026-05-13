@echo off
REM ============================================================
REM REFRESH STAGING VIEWS
REM Location: D:\Dashboard Code\NO_WH\DS\nowhstock\staging\batch\
REM Purpose: Refresh staging materialized views with latest data
REM ============================================================

echo ================================================================================
echo REFRESH STAGING VIEWS
echo ================================================================================
echo.
echo This will refresh:
echo   - mv_slow_fast_moving_summary_staging
echo   - mv_recommendations_complete_staging
echo.
echo WARNING: This may take 5-10 minutes
echo.

cd /d "%~dp0"

echo Refreshing staging views...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "REFRESH MATERIALIZED VIEW mv_slow_fast_moving_summary_staging; REFRESH MATERIALIZED VIEW CONCURRENTLY mv_recommendations_complete_staging; ANALYZE mv_slow_fast_moving_summary_staging; ANALYZE mv_recommendations_complete_staging;"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to refresh staging views!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Staging views refreshed.
echo ================================================================================
echo.
echo View statistics:
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT matviewname, pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size FROM pg_matviews WHERE matviewname LIKE '%_staging' ORDER BY matviewname;"

echo.
pause
