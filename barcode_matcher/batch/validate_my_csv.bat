@echo off
REM ============================================================
REM CSV Format Validator - Check before uploading
REM ============================================================

echo ========================================
echo Barcode CSV Format Validator
echo ========================================
echo.

if "%~1"=="" (
    echo Usage: validate_my_csv.bat "C:\path\to\your\file.csv"
    echo.
    echo Example:
    echo   validate_my_csv.bat "C:\Users\Downloads\mapitemcode.csv"
    echo.
    pause
    exit /b 1
)

cd /d "d:\Dashboard Code\NO_WH\DS\barcode_matcher\batch"
python validate_csv_format.py "%~1"

echo.
pause
