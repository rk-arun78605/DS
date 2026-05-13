@echo off
REM ============================================================
REM Daily Barcode Master Data Loader
REM Loads only NEW entries from barcodedata file
REM ============================================================

echo ========================================
echo Daily Barcode Master Update
echo ========================================
echo.

cd /d "d:\Dashboard Code\NO_WH\DS\barcode_matcher\batch"

echo Running loader...
python load_barcode_master.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Load completed successfully!
) else (
    echo.
    echo ❌ Load failed!
)

echo.
echo Check log: ..\sample_data\barcode_load_log.txt
echo.
pause
