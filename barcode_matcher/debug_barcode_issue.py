"""
Debug script to find why barcodes match in search but not in upload
"""
import psycopg2
import pandas as pd

# Test barcodes
test_barcodes = ['070177109868', '070177067762']

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    database='salesdata'
)

cursor = conn.cursor()

print("=" * 70)
print("CHECKING DATABASE FOR BARCODES")
print("=" * 70)

# Check exact match
for barcode in test_barcodes:
    cursor.execute("""
        SELECT vc_item_barcode, vc_item_code, length(vc_item_barcode) as len 
        FROM barcode_item_master 
        WHERE vc_item_barcode = %s
    """, (barcode,))
    
    result = cursor.fetchone()
    if result:
        print(f"\n✅ FOUND: '{barcode}'")
        print(f"   DB Barcode: '{result[0]}' (length: {result[2]})")
        print(f"   Item Code: '{result[1]}'")
        print(f"   Type: {type(result[0])}")
    else:
        print(f"\n❌ NOT FOUND: '{barcode}'")
        
        # Try without leading zero
        barcode_no_zero = barcode.lstrip('0')
        cursor.execute("""
            SELECT vc_item_barcode, vc_item_code, length(vc_item_barcode) 
            FROM barcode_item_master 
            WHERE vc_item_barcode = %s
        """, (barcode_no_zero,))
        
        result2 = cursor.fetchone()
        if result2:
            print(f"   ⚠️ Found WITHOUT leading zero: '{result2[0]}' -> '{result2[1]}'")

print("\n" + "=" * 70)
print("SIMULATING UPLOAD MATCHING LOGIC")
print("=" * 70)

# Load all barcodes into dict (same as get_barcode_master_cached)
cursor.execute("""
    SELECT vc_item_barcode, vc_item_code 
    FROM barcode_item_master 
    WHERE is_active = TRUE
""")

barcode_dict = {row[0]: row[1] for row in cursor.fetchall()}
print(f"\nTotal barcodes in dict: {len(barcode_dict):,}")

# Test matching with upload logic
print("\nTesting upload matching:")
for barcode in test_barcodes:
    # Simulate Excel read (might convert to float then back to string)
    barcode_as_float = float(barcode)
    barcode_from_excel = str(barcode_as_float)
    
    print(f"\n  Original: '{barcode}'")
    print(f"  As float: {barcode_as_float}")
    print(f"  Back to string: '{barcode_from_excel}'")
    
    # Apply cleaning logic from upload
    barcode_clean = barcode_from_excel.replace('.0', '')
    print(f"  After .0 removal: '{barcode_clean}'")
    
    # Try to match
    item_code = barcode_dict.get(barcode_clean, '')
    if item_code:
        print(f"  ✅ MATCHED -> '{item_code}'")
    else:
        print(f"  ❌ NOT MATCHED")
        
        # Try original without cleaning
        item_code_orig = barcode_dict.get(barcode, '')
        if item_code_orig:
            print(f"  ⚠️ Would match with original: '{item_code_orig}'")

print("\n" + "=" * 70)
print("CHECKING CSV DATA FORMAT")
print("=" * 70)

# Read from CSV to see actual format
import csv
csv_path = r"d:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\barcodedata.csv"

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['VC_ITEM_BARCODE'] in test_barcodes:
            print(f"\nCSV Data:")
            print(f"  Barcode: '{row['VC_ITEM_BARCODE']}' (length: {len(row['VC_ITEM_BARCODE'])})")
            print(f"  Item Code: '{row['VC_ITEM_CODE']}'")

cursor.close()
conn.close()

print("\n" + "=" * 70)
