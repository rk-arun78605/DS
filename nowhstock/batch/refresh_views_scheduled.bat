@echo off
REM ============================================================
REM Scheduled Task: Refresh All Materialized Views
REM ============================================================
REM This batch file is designed to run via Windows Task Scheduler
REM Logs output to refresh_views_log.txt with timestamp
REM ============================================================

set PGPASSWORD=hello
set PSQL="C:\Program Files\PostgreSQL\18\bin\psql.exe"
set LOGFILE=d:\Dashboard Code\NO_WH\DS\refresh_views_log.txt

echo ============================================================ >> "%LOGFILE%"
echo Refresh started at %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

%PSQL% -U postgres -d salesdata -p 3307 -f "d:\Dashboard Code\NO_WH\DS\refresh_all_views.sql" >> "%LOGFILE%" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo ✓ Refresh completed successfully at %DATE% %TIME% >> "%LOGFILE%"
) else (
    echo ✗ Refresh failed with error code %ERRORLEVEL% at %DATE% %TIME% >> "%LOGFILE%"
)

echo. >> "%LOGFILE%"
exit /b %ERRORLEVEL%
