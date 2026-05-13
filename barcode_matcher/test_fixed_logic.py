"""
Test the fixed matching logic
"""
import psycopg2

# Load barcode dict
conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    database='salesdata'
)
cursor = conn.cursor()
cursor.execute("SELECT vc_item_barcode, vc_item_code FROM barcode_item_master WHERE is_active = TRUE")
barcode_dict = {row[0]: row[1] for row in cursor.fetchall()}
cursor.close()
conn.close()

# Test matching function (same as in fixed code)
def match_barcode(barcode_clean):
    # Try exact match first
    item_code = barcode_dict.get(barcode_clean, '')
    if item_code:
        return item_code
    
    # If no match and barcode is numeric, try with leading zero (for UPC-12 -> EAN-13)
    if barcode_clean.isdigit() and len(barcode_clean) == 11:
        # Try adding leading zero for 11-digit codes (UPC without leading zero)
        barcode_with_zero = '0' + barcode_clean
        item_code = barcode_dict.get(barcode_with_zero, '')
        if item_code:
            return item_code
    
    # If still no match and starts with 0, try without leading zeros
    if barcode_clean.startswith('0') and len(barcode_clean) > 11:
        barcode_no_zero = barcode_clean.lstrip('0')
        item_code = barcode_dict.get(barcode_no_zero, '')
        if item_code:
            return item_code
    
    return ''

# Test cases
test_cases = [
    ('070177109868', '070177109868'),  # Original with leading zero
    ('70177109868', '070177109868'),   # Excel stripped leading zero (11 digits)
    ('070177067762', '070177067762'),  # Original with leading zero
    ('70177067762', '070177067762'),   # Excel stripped leading zero (11 digits)
]

print("=" * 70)
print("TESTING FIXED MATCHING LOGIC")
print("=" * 70)

for input_barcode, expected_db_barcode in test_cases:
    print(f"\nInput: '{input_barcode}' (length: {len(input_barcode)})")
    
    item_code = match_barcode(input_barcode)
    
    if item_code:
        print(f"  ✅ MATCHED -> Item Code: '{item_code}'")
        
        # Verify which barcode was matched
        actual_db_barcode = [k for k, v in barcode_dict.items() if v == item_code][0]
        print(f"  📍 Matched DB barcode: '{actual_db_barcode}'")
        
        if actual_db_barcode == expected_db_barcode:
            print(f"  ✓ Correct match!")
        else:
            print(f"  ✗ Expected '{expected_db_barcode}', got '{actual_db_barcode}'")
    else:
        print(f"  ❌ NOT MATCHED")

print("\n" + "=" * 70)
