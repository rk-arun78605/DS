"""
Check what barcodes exist for item code 78427
"""
import psycopg2

conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    database='salesdata'
)

cursor = conn.cursor()

# Check all barcodes for item 78427
cursor.execute("""
    SELECT vc_item_barcode, vc_item_code, length(vc_item_barcode)
    FROM barcode_item_master 
    WHERE vc_item_code = '78427'
    ORDER BY vc_item_barcode
""")

print("All barcodes for item 78427:")
for row in cursor.fetchall():
    print(f"  '{row[0]}' (length: {row[2]})")

# Check if 070177067762 exists
print("\nChecking if 070177067762 exists:")
cursor.execute("""
    SELECT vc_item_barcode, vc_item_code
    FROM barcode_item_master 
    WHERE vc_item_barcode = '070177067762'
""")

result = cursor.fetchone()
if result:
    print(f"  ✅ Found: '{result[0]}' -> Item: '{result[1]}'")
else:
    print(f"  ❌ NOT found in database")

# Check with LIKE for similar barcodes
print("\nSimilar barcodes containing '70177067762':")
cursor.execute("""
    SELECT vc_item_barcode, vc_item_code, length(vc_item_barcode)
    FROM barcode_item_master 
    WHERE vc_item_barcode LIKE '%70177067762%'
""")

for row in cursor.fetchall():
    print(f"  '{row[0]}' -> '{row[1]}' (length: {row[2]})")

cursor.close()
conn.close()
