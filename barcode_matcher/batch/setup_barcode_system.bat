@echo off
REM ============================================================
REM Setup Barcode Matching System
REM ============================================================

echo ========================================
echo Barcode Matcher Setup
echo ========================================
echo.

REM Set PostgreSQL path
set PSQL="C:\Program Files\PostgreSQL\18\bin\psql.exe"

echo [1/2] Creating database tables and functions...
%PSQL% -U postgres -p 3307 -d salesdata -f "..\sql\create_barcode_tables.sql"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Database setup completed successfully!
    echo.
) else (
    echo.
    echo ❌ Database setup failed!
    echo.
    pause
    exit /b 1
)

echo [2/2] Verifying setup...
%PSQL% -U postgres -p 3307 -d salesdata -c "SELECT COUNT(*) AS total_barcodes FROM barcode_item_master WHERE is_active = TRUE;"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run: run_barcode_matcher.bat
echo 2. Upload your barcode file
echo 3. Download results with item codes
echo.
pause
