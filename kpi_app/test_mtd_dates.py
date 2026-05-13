"""
Quick test to see what MTD dates are being used
"""
from datetime import datetime, timedelta

def get_mtd_dates():
    """Get MTD date range (1st of current month to yesterday)"""
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    start_date = yesterday.replace(day=1)
    return start_date.date(), yesterday.date()

start_date, end_date = get_mtd_dates()
print(f"Today: {datetime.today().date()}")
print(f"MTD Start (2025): {start_date}")
print(f"MTD End (2025): {end_date}")
print(f"MTD Start (2024): {start_date.replace(year=2024)}")
print(f"MTD End (2024): {end_date.replace(year=2024)}")

# Now test the actual query
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    database='salesdata'
)

query_2024 = """
SELECT 
    SUM("QTY") as qty,
    SUM("NET_SALES") as net_sales,
    COUNT(*) as row_count
FROM sales_2024
WHERE "DATE_INVOICE"::date >= %s 
  AND "DATE_INVOICE"::date <= %s
"""

query_2025 = """
SELECT 
    SUM("QTY") as qty,
    SUM("NET_SALES") as net_sales,
    COUNT(*) as row_count
FROM sales_2025
WHERE "DATE_INVOICE"::date >= %s 
  AND "DATE_INVOICE"::date <= %s
"""

print("\n" + "="*60)
print("TESTING MTD QUERIES")
print("="*60)

# Test 2024 query
df_2024 = pd.read_sql(query_2024, conn, params=(
    start_date.replace(year=2024),
    end_date.replace(year=2024)
))
print(f"\n2024 MTD ({start_date.replace(year=2024)} to {end_date.replace(year=2024)}):")
print(df_2024.to_string())

# Test 2025 query
df_2025 = pd.read_sql(query_2025, conn, params=(
    start_date,
    end_date
))
print(f"\n2025 MTD ({start_date} to {end_date}):")
print(df_2025.to_string())

conn.close()

print("\n" + "="*60)
if df_2024['net_sales'][0] > 0:
    print("✅ 2024 data EXISTS")
else:
    print("❌ 2024 data is ZERO or NULL")
    
if df_2025['net_sales'][0] > 0:
    print("✅ 2025 data EXISTS")
else:
    print("❌ 2025 data is ZERO or NULL")
print("="*60)
