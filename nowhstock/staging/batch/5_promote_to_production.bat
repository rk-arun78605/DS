@echo off
REM ============================================================
REM PROMOTE STAGING TO PRODUCTION
REM Location: D:\Dashboard Code\NO_WH\DS\nowhstock\staging\batch\
REM Purpose: Replace production views with staging (AFTER APPROVAL)
REM ============================================================

echo ================================================================================
echo PROMOTE STAGING TO PRODUCTION
echo ================================================================================
echo.
echo WARNING: This will REPLACE production with staging!
echo.
echo Steps:
echo   1. Backup current production view
echo   2. Drop production view
echo   3. Rename staging view to production
echo   4. Rename all indexes
echo   5. Update production dashboard code
echo.
echo IMPORTANT: Make sure you have tested staging thoroughly!
echo.
set /p confirm="Are you sure you want to promote staging to production? (YES/no): "

if /i NOT "%confirm%"=="YES" (
    echo.
    echo Promotion cancelled.
    pause
    exit /b 0
)

echo.
echo ================================================================================
echo STEP 1: Backup Production View
echo ================================================================================
echo.

cd /d "%~dp0"

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "CREATE TABLE mv_recommendations_complete_backup_$(date +%%Y%%m%%d) AS SELECT * FROM mv_recommendations_complete;"

if errorlevel 1 (
    echo ERROR: Failed to backup production view!
    pause
    exit /b 1
)

echo Backup created successfully.
echo.
echo ================================================================================
echo STEP 2: Drop Production View
echo ================================================================================
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "DROP MATERIALIZED VIEW IF EXISTS mv_recommendations_complete CASCADE;"

echo.
echo ================================================================================
echo STEP 3: Rename Staging to Production
echo ================================================================================
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "ALTER MATERIALIZED VIEW mv_recommendations_complete_staging RENAME TO mv_recommendations_complete;"

echo.
echo ================================================================================
echo STEP 4: Rename Indexes
echo ================================================================================
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "ALTER INDEX idx_mv_recs_staging_item_dest RENAME TO idx_mv_recs_complete_item_dest; ALTER INDEX idx_mv_recs_staging_source RENAME TO idx_mv_recs_complete_source; ALTER INDEX idx_mv_recs_staging_dest RENAME TO idx_mv_recs_complete_dest; ALTER INDEX idx_mv_recs_staging_item RENAME TO idx_mv_recs_complete_item; ALTER INDEX idx_mv_recs_staging_qty RENAME TO idx_mv_recs_complete_qty; ALTER INDEX idx_mv_recs_staging_groups RENAME TO idx_mv_recs_complete_groups; ALTER INDEX idx_mv_recs_staging_subgroup RENAME TO idx_mv_recs_complete_subgroup;"

echo.
echo ================================================================================
echo STEP 5: Analyze New Production View
echo ================================================================================
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "ANALYZE mv_recommendations_complete;"

echo.
echo ================================================================================
echo STEP 6: Update Production SQL File
echo ================================================================================
echo.

echo Copying staging SQL to production location...
copy /Y "..\sql\create_staging_views.sql" "..\..\..\NowhStock_mv_recommendations_complete.sql"

echo.
echo ================================================================================
echo SUCCESS! STAGING PROMOTED TO PRODUCTION
echo ================================================================================
echo.
echo What changed:
echo   - Production now uses priority-to-priority transfer logic
echo   - Priority shops can now be sources
echo   - Source != destination enforced
echo.
echo Next steps:
echo   1. Test production dashboard (port 8501)
echo   2. Verify recommendations look correct
echo   3. Update refresh_all_views.sql if needed
echo   4. Inform users of the change
echo.
echo Production dashboard: streamlit run d:\Dashboard Code\NO_WH\DS\nowhstock_ds.py
echo.
pause
