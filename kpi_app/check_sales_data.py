"""
Check sales_2024 and sales_2025 table data
Run this to diagnose why 2024 numbers are not showing
"""
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

# Database connection
conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    database='salesdata'
)

print("=" * 80)
print("SALES TABLE DIAGNOSTICS")
print("=" * 80)

# 1. Check table structure
print("\n1. CHECKING TABLE COLUMNS...")
query = """
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name IN ('sales_2024', 'sales_2025')
  AND table_schema = 'public'
ORDER BY table_name, ordinal_position;
"""
df_cols = pd.read_sql(query, conn)
print(df_cols.to_string())

# 2. Check data counts
print("\n2. CHECKING ROW COUNTS...")
query_2024 = "SELECT COUNT(*) as count FROM sales_2024;"
query_2025 = "SELECT COUNT(*) as count FROM sales_2025;"
count_2024 = pd.read_sql(query_2024, conn)['count'][0]
count_2025 = pd.read_sql(query_2025, conn)['count'][0]
print(f"sales_2024: {count_2024:,} rows")
print(f"sales_2025: {count_2025:,} rows")

# 3. Check date ranges
print("\n3. CHECKING DATE RANGES...")
try:
    query = """
    SELECT 
        MIN("DATE_INVOICE") as min_date,
        MAX("DATE_INVOICE") as max_date,
        COUNT(*) as total_rows,
        SUM("NET_SALES") as total_sales
    FROM sales_2024;
    """
    df_2024 = pd.read_sql(query, conn)
    print("sales_2024:")
    print(df_2024.to_string())
except Exception as e:
    print(f"ERROR querying sales_2024: {e}")

try:
    query = """
    SELECT 
        MIN("DATE_INVOICE") as min_date,
        MAX("DATE_INVOICE") as max_date,
        COUNT(*) as total_rows,
        SUM("NET_SALES") as total_sales
    FROM sales_2025;
    """
    df_2025 = pd.read_sql(query, conn)
    print("\nsales_2025:")
    print(df_2025.to_string())
except Exception as e:
    print(f"ERROR querying sales_2025: {e}")

# 4. Check MTD data (January 1 to Dec 3)
print("\n4. CHECKING MTD DATA (Jan 1 to Dec 3)...")
mtd_start_2024 = datetime(2024, 1, 1).date()
mtd_end_2024 = datetime(2024, 12, 3).date()

try:
    query = """
    SELECT 
        COUNT(*) as row_count,
        SUM("NET_SALES") as total_sales,
        SUM("QTY") as total_qty
    FROM sales_2024
    WHERE "DATE_INVOICE"::date >= %s 
      AND "DATE_INVOICE"::date <= %s;
    """
    df_mtd_2024 = pd.read_sql(query, conn, params=(mtd_start_2024, mtd_end_2024))
    print(f"sales_2024 MTD ({mtd_start_2024} to {mtd_end_2024}):")
    print(df_mtd_2024.to_string())
except Exception as e:
    print(f"ERROR: {e}")

mtd_start_2025 = datetime(2025, 1, 1).date()
mtd_end_2025 = datetime(2025, 12, 3).date()

try:
    query = """
    SELECT 
        COUNT(*) as row_count,
        SUM("NET_SALES") as total_sales,
        SUM("QTY") as total_qty
    FROM sales_2025
    WHERE "DATE_INVOICE"::date >= %s 
      AND "DATE_INVOICE"::date <= %s;
    """
    df_mtd_2025 = pd.read_sql(query, conn, params=(mtd_start_2025, mtd_end_2025))
    print(f"\nsales_2025 MTD ({mtd_start_2025} to {mtd_end_2025}):")
    print(df_mtd_2025.to_string())
except Exception as e:
    print(f"ERROR: {e}")

# 5. Check top departments
print("\n5. CHECKING TOP 5 DEPARTMENTS BY SALES...")
try:
    query = """
    SELECT 
        "DEPT",
        SUM("NET_SALES") as total_sales,
        SUM("QTY") as total_qty
    FROM sales_2024
    GROUP BY "DEPT"
    ORDER BY total_sales DESC
    LIMIT 5;
    """
    df_top_2024 = pd.read_sql(query, conn)
    print("sales_2024 Top 5 Departments:")
    print(df_top_2024.to_string())
except Exception as e:
    print(f"ERROR: {e}")

try:
    query = """
    SELECT 
        "DEPT",
        SUM("NET_SALES") as total_sales,
        SUM("QTY") as total_qty
    FROM sales_2025
    GROUP BY "DEPT"
    ORDER BY total_sales DESC
    LIMIT 5;
    """
    df_top_2025 = pd.read_sql(query, conn)
    print("\nsales_2025 Top 5 Departments:")
    print(df_top_2025.to_string())
except Exception as e:
    print(f"ERROR: {e}")

# 6. Check existing indexes
print("\n6. CHECKING EXISTING INDEXES...")
query = """
SELECT 
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE tablename IN ('sales_2024', 'sales_2025')
ORDER BY tablename, indexname;
"""
df_indexes = pd.read_sql(query, conn)
print(df_indexes.to_string())

# 7. Check sample data
print("\n7. SAMPLE DATA FROM sales_2024 (first 3 rows)...")
try:
    query = 'SELECT * FROM sales_2024 LIMIT 3;'
    df_sample = pd.read_sql(query, conn)
    print(df_sample.transpose().to_string())
except Exception as e:
    print(f"ERROR: {e}")

conn.close()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print("\nNext Steps:")
print("1. If sales_2024 has no data or wrong date range, you need to load 2024 data")
print("2. If column names are different, update the dashboard queries")
print("3. Run OPTIMIZE_KPI_DASHBOARD.sql to create indexes and materialized views")
print("=" * 80)
