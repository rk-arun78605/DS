#!/usr/bin/env python3
"""Direct test of load_offloading_vs_loading function logic."""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'dbname': 'WH'
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def test_load_function():
    """Test the exact load_offloading_vs_loading function."""
    start_date = datetime(2026, 3, 1)
    end_date = datetime(2026, 4, 7)
    
    diff_val_expr = "((COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) * COALESCE(price, 0))"
    query = f"""
        SELECT
            date,
            shop_code,
            vehicle_no,
            item_code,
            item_name,
            COALESCE(qty_loaded, 0)::numeric AS qty_loaded,
            (COALESCE(qty_loaded, 0) * COALESCE(price, 0))::numeric AS value_loaded,
            COALESCE(qty_offloaded, 0)::numeric AS qty_offloaded,
            (COALESCE(qty_offloaded, 0) * COALESCE(price, 0))::numeric AS value_offloaded,
            (COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0))::numeric AS diff_qty,
            ({diff_val_expr})::numeric AS diff_val
        FROM offloading_vs_loading
        WHERE date BETWEEN %(s)s AND %(e)s
        AND item_code = '06587'
        ORDER BY date DESC, shop_code, vehicle_no, item_code
    """
    
    print("🔍 Testing load_offloading_vs_loading() for item 06587:")
    print(f"   Date range: {start_date.date()} to {end_date.date()}\n")
    
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, {"s": start_date, "e": end_date})
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        
        if not rows:
            print("  ❌ No rows returned!")
            return
        
        print("✅ Query Results (first 3 rows):\n")
        for i, r in enumerate(rows[:3]):
            print(f"  Row {i+1}:")
            print(f"    Item Code: {r['item_code']}")
            print(f"    Date: {r['date']}")
            print(f"    Shop: {r['shop_code']}")
            print(f"    QTY Loaded: {r['qty_loaded']}")
            print(f"    VALUE Loaded: {r['value_loaded']} (from query: {r['qty_loaded']} × price)")
            print(f"    QTY Offloaded: {r['qty_offloaded']}")
            print(f"    VALUE Offloaded: {r['value_offloaded']} (from query: {r['qty_offloaded']} × price)")
            print(f"    Diff QTY: {r['diff_qty']}")
            print(f"    Diff Value: {r['diff_val']} (from query: ({r['qty_offloaded']} - {r['qty_loaded']}) × price)")
            print()
        
        print(f"✅ Total rows found: {len(rows)}")
        
    except psycopg2.Error as e:
        print(f"❌ Database Error: {e}")

if __name__ == '__main__':
    test_load_function()
