#!/usr/bin/env python3
"""Verify dashboard query for item 06587."""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'dbname': 'WH'
}

def test_dashboard_query():
    """Test the exact dashboard query logic."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Use a date range that includes all item 06587 rows
        start_date = datetime(2026, 1, 1)
        end_date = datetime(2026, 4, 7)
        
        diff_val_expr = "((COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) * COALESCE(price, 0))"
        query = f"""
            SELECT
                date,
                shop_code,
                vehicle_no,
                item_code,
                item_name,
                price,
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
        
        print("🔍 Testing dashboard query for item 06587:")
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, {"s": start_date, "e": end_date})
            rows = cur.fetchall()
            
            if not rows:
                print("  ❌ No rows found")
            else:
                for i, row in enumerate(rows):
                    print(f"\n  Row {i+1}:")
                    print(f"    Item: {row['item_code']}, Date: {row['date']}, Shop: {row['shop_code']}")
                    print(f"    Qty Loaded: {row['qty_loaded']}, Qty Offloaded: {row['qty_offloaded']}, Price: {row['price']}")
                    print(f"    ✓ WH Loaded Value: {row['qty_loaded']} × {row['price']} = {row['value_loaded']}")
                    print(f"    ✓ Shop Receiving Value: {row['qty_offloaded']} × {row['price']} = {row['value_offloaded']}")
                    print(f"    ✓ Diff Value: ({row['qty_offloaded']} - {row['qty_loaded']}) × {row['price']} = {row['diff_val']}")
        
        conn.close()
        print("\n✅ Dashboard query test complete!")
    except psycopg2.Error as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    test_dashboard_query()
