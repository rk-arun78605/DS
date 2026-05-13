import psycopg2
import pandas as pd

conn = psycopg2.connect(host='localhost', port=3307, user='postgres', password='hello', database='salesdata')

# Check what December 2024 dates actually exist
query = """
SELECT 
    "DATE_INVOICE"::date as sale_date,
    COUNT(*) as row_count,
    SUM("NET_SALES") as total_sales
FROM sales_2024
WHERE "DATE_INVOICE"::date >= '2024-12-01' 
  AND "DATE_INVOICE"::date <= '2024-12-10'
GROUP BY "DATE_INVOICE"::date
ORDER BY "DATE_INVOICE"::date;
"""

df = pd.read_sql(query, conn)
print("December 2024 dates in sales_2024:")
print(df.to_string())

if df.empty:
    print("\n❌ NO DECEMBER 2024 DATA FOUND!")
    print("\nChecking what the last dates are in sales_2024:")
    query2 = """
    SELECT "DATE_INVOICE"::date as sale_date, COUNT(*) as rows
    FROM sales_2024
    GROUP BY "DATE_INVOICE"::date
    ORDER BY "DATE_INVOICE"::date DESC
    LIMIT 10;
    """
    df2 = pd.read_sql(query2, conn)
    print(df2.to_string())

conn.close()
