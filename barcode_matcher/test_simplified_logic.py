"""
Test the simplified matching logic
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

# Test matching function (simplified - only adds leading zero for 11-digit codes)
def match_barcode(barcode_clean):
    # Try exact match first
    item_code = barcode_dict.get(barcode_clean, '')
    if item_code:
        return item_code
    
    # If no match and barcode is numeric, try with leading zero (for UPC-12 -> EAN-13)
    if barcode_clean.isdigit() and len(barcode_clean) == 11:
        # Try adding single leading zero for 11-digit codes (UPC without leading zero)
        barcode_with_zero = '0' + barcode_clean
        item_code = barcode_dict.get(barcode_with_zero, '')
        if item_code:
            return item_code
    
    return ''

# Test cases
test_cases = [
    ('070177109868', 'Expected: 78430'),  # Original 12-digit
    ('70177109868', 'Expected: 78430'),   # Excel stripped to 11 digits
    ('070177067762', 'Expected: 78427'),  # Original 12-digit
    ('70177067762', 'Expected: 78427'),   # Excel stripped to 11 digits
]

print("=" * 80)
print("TESTING SIMPLIFIED MATCHING LOGIC (Only adds leading 0 for 11-digit codes)")
print("=" * 80)

for input_barcode, expected in test_cases:
    print(f"\nInput: '{input_barcode}' (length: {len(input_barcode)}) | {expected}")
    
    item_code = match_barcode(input_barcode)
    
    if item_code:
        print(f"  ✅ MATCHED -> Item Code: '{item_code}'")
        
        # Find which DB barcode was matched
        matched_barcodes = [k for k, v in barcode_dict.items() if v == item_code and k in [input_barcode, '0' + input_barcode]]
        if matched_barcodes:
            print(f"  📍 Matched DB barcode: '{matched_barcodes[0]}'")
    else:
        print(f"  ❌ NOT MATCHED")

print("\n" + "=" * 80)
