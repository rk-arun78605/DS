@echo off
REM Quick check if staging view exists and its count
echo Checking staging view status...
echo.

set PGPASSWORD=hello
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT 'Staging View' as view, COUNT(*) as count FROM mv_recommendations_complete_staging UNION ALL SELECT 'Production View' as view, COUNT(*) as count FROM mv_recommendations_complete;"

if errorlevel 1 (
    echo.
    echo Staging view does NOT exist yet.
    echo Run: 1_setup_staging.bat to create it.
) else (
    echo.
    echo Staging view EXISTS and is ready!
)

pause
