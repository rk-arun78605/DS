import psycopg2

conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    database='WH'
)

cur = conn.cursor()
serial = 'AKAI-CF073AC-99L-25C10106'

# Search for similar patterns
print('=' * 70)
print(f'Searching for serials similar to: {serial}')
print('=' * 70)

# Search in WH for AKAI items
print('\nWH Receiving - AKAI serials:')
cur.execute("""
SELECT item_code, serial_no, grn_date, COUNT(*) 
FROM whreceived_serialno
WHERE item_code LIKE '%AKAI%' OR serial_no LIKE '%AKAI%'
GROUP BY item_code, serial_no, grn_date
LIMIT 10
""")
results = cur.fetchall()
if results:
    for row in results:
        print(f"  {row}")
else:
    print("  No AKAI serials found in WH")

# Search in Shop Selling for AKAI items
print('\nShop Selling - AKAI serials with serial_check=N:')
cur.execute("""
SELECT item_code, serial_number, serial_check, bill_date
FROM serialno_check_yes_no
WHERE (item_code LIKE '%AKAI%' OR serial_number LIKE '%AKAI%') 
  AND serial_check = 'N'
LIMIT 10
""")
results = cur.fetchall()
if results:
    for row in results:
        print(f"  {row}")
else:
    print("  No AKAI serials with status='N' found in Shop Selling")

# Search for the specific item code part
print('\nSearching in WH for partial match (CF073AC):')
cur.execute("""
SELECT DISTINCT item_code, serial_no
FROM whreceived_serialno
WHERE serial_no LIKE '%CF073AC%' OR item_code LIKE '%CF073AC%'
LIMIT 5
""")
results = cur.fetchall()
if results:
    for row in results:
        print(f"  Item: {row[0]}, Serial: {row[1]}")
else:
    print("  No matches found")

print('\nSearching in Shop Selling for partial match (CF073AC):')
cur.execute("""
SELECT DISTINCT item_code, serial_number, serial_check
FROM serialno_check_yes_no
WHERE serial_number LIKE '%CF073AC%' OR item_code LIKE '%CF073AC%'
LIMIT 5
""")
results = cur.fetchall()
if results:
    for row in results:
        print(f"  Item: {row[0]}, Serial: {row[1]}, Check: {row[2]}")
else:
    print("  No matches found")

conn.close()
print('\n' + '=' * 70)
print('Serial not found. Please verify the serial number is correct.')
print('=' * 70)
