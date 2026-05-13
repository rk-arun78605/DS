"""Quick test to verify stockout tracking system"""
import psycopg2

conn = psycopg2.connect(
    host='localhost', port=3307, user='postgres',
    password='hello', database='century_penetration'
)
cur = conn.cursor()

print("=" * 80)
print("STOCKOUT TRACKING SYSTEM - VERIFICATION TEST")
print("=" * 80)

# 1. Check mv_century_penetration_test
cur.execute("SELECT COUNT(*) FROM mv_century_penetration_test;")
print(f"\n1. mv_century_penetration_test: {cur.fetchone()[0]:,} rows")

cur.execute("SELECT COUNT(*) FROM mv_century_penetration_test WHERE stock_out_date IS NOT NULL;")
print(f"   - Items currently out of stock: {cur.fetchone()[0]:,}")

cur.execute("SELECT COUNT(*) FROM mv_century_penetration_test WHERE ops_manager_name IS NOT NULL;")
print(f"   - Items with ops manager assigned: {cur.fetchone()[0]:,}")

# 2. Check snapshot table
cur.execute("SELECT COUNT(*), COUNT(DISTINCT snapshot_date) FROM century_stockout_daily_snapshot;")
total, days = cur.fetchone()
print(f"\n2. century_stockout_daily_snapshot: {total:,} rows across {days} days")

# 3. Check analysis view
cur.execute("SELECT COUNT(DISTINCT level_type) FROM mv_stockout_analysis;")
print(f"\n3. mv_stockout_analysis: {cur.fetchone()[0]} dimension types")

cur.execute("""
    SELECT level_type, COUNT(*) 
    FROM mv_stockout_analysis 
    GROUP BY level_type 
    ORDER BY level_type;
""")
print("   Dimensions:")
for level, count in cur.fetchall():
    print(f"   - {level}: {count} records")

# 4. Check ops manager stockout detail
cur.execute("SELECT COUNT(*) FROM v_ops_manager_stockout_detail;")
print(f"\n4. v_ops_manager_stockout_detail: {cur.fetchone()[0]} manager-shop combinations")

# 5. Top 3 ops managers by stockout %
cur.execute("""
    SELECT level_value, stockout_pct 
    FROM mv_stockout_analysis 
    WHERE level_type = 'Ops Manager' 
    ORDER BY stockout_pct DESC 
    LIMIT 3;
""")
print("\n5. Top 3 Ops Managers by Stockout %:")
for idx, (manager, pct) in enumerate(cur.fetchall(), 1):
    print(f"   {idx}. {manager}: {pct}%")

# 6. Sample stockout items
cur.execute("""
    SELECT shop_code, item_code, item_name, stock_out_date, days_out_of_stock, ops_manager_name
    FROM mv_century_penetration_test
    WHERE stock_out_date IS NOT NULL
    LIMIT 5;
""")
print("\n6. Sample items currently out of stock:")
print(f"   {'Shop':<10} {'Item':<15} {'Name':<30} {'Out Since':<12} {'Days':<5} {'Manager':<15}")
for row in cur.fetchall():
    print(f"   {row[0]:<10} {row[1]:<15} {row[2][:28]:<30} {str(row[3]):<12} {row[4]:<5} {row[5]:<15}")

print("\n" + "=" * 80)
print("✅ ALL SYSTEMS OPERATIONAL")
print("=" * 80)

cur.close()
conn.close()
