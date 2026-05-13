import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd

conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    dbname='century_penetration'
)

cursor = conn.cursor(cursor_factory=RealDictCursor)

# First check columns
cursor.execute("""
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'century_stockout_daily_snapshot' 
ORDER BY ordinal_position
""")
print("Snapshot Table Columns:", [r['column_name'] for r in cursor.fetchall()])

# Check daily snapshot data for last 10 days
query = """
SELECT 
    snapshot_date,
    COUNT(*) as total_items,
    SUM(CASE WHEN is_out_of_stock = TRUE THEN 1 ELSE 0 END) as stockout_items,
    ROUND(100.0 * SUM(CASE WHEN is_out_of_stock = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) as stockout_pct
FROM century_stockout_daily_snapshot
GROUP BY snapshot_date
ORDER BY snapshot_date DESC
LIMIT 10;
"""

cursor.execute(query)
df = pd.DataFrame(cursor.fetchall())
print("\nDaily Stockout Trend Data:")
print(df)

cursor.close()
conn.close()
