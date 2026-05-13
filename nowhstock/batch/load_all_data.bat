@echo off
REM ============================================================
REM LOAD ALL DATA - Batch File Runner with Logging
REM ============================================================
REM This batch file runs the data loading SQL script with comprehensive logging.
REM It loads data from CSV files into all PostgreSQL tables.
REM 
REM Prerequisites:
REM - CSV files must exist in D:/Dashboard Code/tbl_data/ subdirectories
REM - PostgreSQL server must be running on localhost:3307
REM - Tables must already exist (schema must be created)
REM 
REM Usage:
REM   Double-click this file or run: load_all_data.bat
REM ============================================================

SETLOCAL ENABLEDELAYEDEXPANSION

REM Set working directory
cd /d "d:\Dashboard Code\NO_WH\DS"

REM PostgreSQL configuration
set PGHOST=localhost
set PGPORT=3307
set PGUSER=postgres
set PGPASSWORD=hello
set PGDATABASE=salesdata

REM File paths
set SQL_SCRIPT=load_all_data.sql
set LOG_FILE=data_load_log.txt
set PSQL="C:\Program Files\PostgreSQL\18\bin\psql.exe"

REM Check if psql exists
if not exist %PSQL% (
    echo ============================================================
    echo ERROR: PostgreSQL psql.exe not found!
    echo Expected location: %PSQL%
    echo.
    echo Please update the PSQL variable in this batch file with the correct path.
    echo ============================================================
    pause
    exit /b 1
)

REM Check if SQL script exists
if not exist "%SQL_SCRIPT%" (
    echo ============================================================
    echo ERROR: SQL script not found: %SQL_SCRIPT%
    echo Make sure you are running this from the correct directory.
    echo ============================================================
    pause
    exit /b 1
)

REM Print header
echo.
echo ============================================================
echo LOAD ALL DATA - Starting Data Import
echo ============================================================
echo.
echo Database: %PGDATABASE%
echo Host: %PGHOST%:%PGPORT%
echo User: %PGUSER%
echo Script: %SQL_SCRIPT%
echo Log File: %LOG_FILE%
echo.
echo This process will:
echo   1. Load nowhstock data
echo   2. Load supplier shop GRN data
echo   3. Load item details
echo   4. Load shop expiry data
echo   5. Load WH GRN details
echo   6. Load stock in transit (SIT) data
echo   7. Load sales 2025 data (LARGEST - may take 5-15 minutes)
echo.
echo WARNING: This will TRUNCATE and reload all tables!
echo.
pause

REM Log start time
echo ============================================================ > "%LOG_FILE%"
echo DATA LOAD STARTED AT %DATE% %TIME% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Run the SQL script
echo.
echo Running SQL script... (check %LOG_FILE% for details)
echo.

%PSQL% -h %PGHOST% -p %PGPORT% -U %PGUSER% -d %PGDATABASE% -f "%SQL_SCRIPT%" >> "%LOG_FILE%" 2>&1

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

REM Log completion
echo. >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
if %EXIT_CODE% EQU 0 (
    echo DATA LOAD COMPLETED SUCCESSFULLY AT %DATE% %TIME% >> "%LOG_FILE%"
    echo Exit Code: %EXIT_CODE% >> "%LOG_FILE%"
    echo ============================================================ >> "%LOG_FILE%"
    echo.
    echo ============================================================
    echo SUCCESS - All data loaded successfully!
    echo ============================================================
    echo.
    echo Log file: %LOG_FILE%
    echo.
    echo Next steps:
    echo   1. Refresh materialized views: python refresh_materialized_views.py
    echo   2. Restart Streamlit dashboard: streamlit run nowhstock_ds.py
    echo.
) else (
    echo DATA LOAD FAILED AT %DATE% %TIME% >> "%LOG_FILE%"
    echo Exit Code: %EXIT_CODE% >> "%LOG_FILE%"
    echo ============================================================ >> "%LOG_FILE%"
    echo.
    echo ============================================================
    echo ERROR - Data load failed!
    echo ============================================================
    echo.
    echo Exit Code: %EXIT_CODE%
    echo Check log file for details: %LOG_FILE%
    echo.
)

echo Press any key to exit...
pause > nul

exit /b %EXIT_CODE%
