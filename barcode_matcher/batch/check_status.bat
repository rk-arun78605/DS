@echo off
REM Quick status check for barcode database
echo ===============================================
echo Barcode Database Status
echo ===============================================
echo.

"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT COUNT(*) AS total_barcodes FROM barcode_item_master WHERE is_active = TRUE;"

echo.
echo Last 5 entries:
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT vc_item_barcode, vc_item_code, created_at FROM barcode_item_master ORDER BY created_at DESC LIMIT 5;"

echo.
pause
