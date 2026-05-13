import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    database='WH'
)

serial = 'AKAI-CF073AC-99L-25C10106'

# Check WH Received
print('=' * 60)
print('WH RECEIVED TABLE (whreceived_serialno)')
print('=' * 60)
wh_query = f"""
SELECT 
    item_code,
    serial_no,
    grn_date,
    COUNT(*) as count
FROM whreceived_serialno
WHERE TRIM(serial_no) = '{serial}'
GROUP BY item_code, serial_no, grn_date
"""
wh_df = pd.read_sql(wh_query, conn)
if not wh_df.empty:
    print(wh_df.to_string(index=False))
else:
    print('❌ NO RECORDS FOUND')

# Check Shop Selling
print('\n' + '=' * 60)
print('SHOP SELLING TABLE (serialno_check_yes_no)')
print('=' * 60)
ss_query = f"""
SELECT 
    item_code,
    serial_number,
    serial_check,
    bill_date,
    COUNT(*) as count
FROM serialno_check_yes_no
WHERE TRIM(serial_number) = '{serial}'
GROUP BY item_code, serial_number, serial_check, bill_date
"""
ss_df = pd.read_sql(ss_query, conn)
if not ss_df.empty:
    print(ss_df.to_string(index=False))
else:
    print('❌ NO RECORDS FOUND')

# Summary
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
if not wh_df.empty:
    print(f"✅ Serial EXISTS in WH Receiving: {len(wh_df)} records")
else:
    print(f"❌ Serial NOT found in WH Receiving")

if not ss_df.empty:
    print(f"✅ Serial EXISTS in Shop Selling: {len(ss_df)} records")
    for _, row in ss_df.iterrows():
        print(f"   - Status: {row['serial_check']} (Y=Found, N=Not Found)")
        print(f"   - Item: {row['item_code']}")
        print(f"   - Bill Date: {row['bill_date']}")
else:
    print(f"❌ Serial NOT found in Shop Selling")

conn.close()
