@echo off
REM ============================================================
REM Setup Barcode Matching System - UPDATED
REM ============================================================

echo ========================================
echo Barcode Matcher Setup
echo ========================================
echo.

REM Set PostgreSQL path
set PSQL="C:\Program Files\PostgreSQL\18\bin\psql.exe"

echo [1/3] Creating database tables and functions...
%PSQL% -U postgres -p 3307 -d salesdata -f "..\sql\create_barcode_tables_clean.sql"

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

echo [2/3] Loading initial barcode data from barcodedata file...
python load_barcode_master.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Initial data loaded successfully!
    echo.
) else (
    echo.
    echo ❌ Data load failed!
    echo.
)

echo [3/3] Verifying setup...
%PSQL% -U postgres -p 3307 -d salesdata -c "SELECT COUNT(*) AS total_barcodes FROM barcode_item_master WHERE is_active = TRUE;"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Master data location:
echo   D:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\barcodedata.*
echo.
echo Daily update: Run load_barcode_master.bat
echo.
echo Next steps:
echo 1. Run: run_barcode_matcher.bat
echo 2. Upload CSV with single 'barcode' column
echo 3. Download results with item codes
echo.
pause
