@echo off
REM ============================================================
REM COMPARE STAGING VS PRODUCTION
REM Location: D:\Dashboard Code\NO_WH\DS\nowhstock\staging\batch\
REM Purpose: Compare staging and production recommendations
REM ============================================================

echo ================================================================================
echo STAGING VS PRODUCTION COMPARISON
echo ================================================================================
echo.

cd /d "%~dp0"

echo [1/5] Total Recommendations...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT 'Production' as env, COUNT(*) as total_recs, COUNT(DISTINCT item_code) as unique_items, COUNT(DISTINCT source_shop) as unique_sources FROM mv_recommendations_complete UNION ALL SELECT 'Staging' as env, COUNT(*) as total_recs, COUNT(DISTINCT item_code) as unique_items, COUNT(DISTINCT source_shop) as unique_sources FROM mv_recommendations_complete_staging;"

echo.
echo [2/5] Priority Shop Sources...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT 'Production' as env, COUNT(DISTINCT source_shop) as priority_sources FROM mv_recommendations_complete WHERE source_shop IN ('SPN','MSS','LFS','M03','KAS','MM1','MM2','FAR','KS7','WHL','MM3') UNION ALL SELECT 'Staging' as env, COUNT(DISTINCT source_shop) as priority_sources FROM mv_recommendations_complete_staging WHERE source_shop IN ('SPN','MSS','LFS','M03','KAS','MM1','MM2','FAR','KS7','WHL','MM3');"

echo.
echo [3/5] Same-Shop Transfers (should be 0)...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT 'Production' as env, COUNT(*) as same_shop FROM mv_recommendations_complete WHERE source_shop = dest_shop UNION ALL SELECT 'Staging' as env, COUNT(*) as same_shop FROM mv_recommendations_complete_staging WHERE source_shop = dest_shop;"

echo.
echo [4/5] Top 10 Priority-to-Priority Transfers (Staging Only)...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT source_shop, dest_shop, COUNT(*) as transfer_count, SUM(recommended_qty) as total_qty FROM mv_recommendations_complete_staging WHERE source_shop IN ('SPN','MSS','LFS','M03','KAS','MM1','MM2','FAR','KS7','WHL','MM3') GROUP BY source_shop, dest_shop ORDER BY SUM(recommended_qty) DESC LIMIT 10;"

echo.
echo [5/5] View Sizes...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT matviewname, pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size FROM pg_matviews WHERE matviewname IN ('mv_recommendations_complete', 'mv_recommendations_complete_staging') ORDER BY matviewname;"

echo.
echo ================================================================================
echo COMPARISON COMPLETE
echo ================================================================================
echo.
echo KEY METRICS TO VALIDATE:
echo   1. Staging should have MORE recommendations than production
echo   2. Staging should have priority shop sources (production has 0)
echo   3. Same-shop transfers should be 0 in BOTH
echo   4. Priority-to-priority transfers should appear in staging only
echo.
pause
