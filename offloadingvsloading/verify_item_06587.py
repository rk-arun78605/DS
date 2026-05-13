#!/usr/bin/env python3
"""Verify item code 06587 values."""
import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'dbname': 'WH'
}

def verify_item():
    """Check item 06587 values in both table and MV."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Check base table
        print("📊 Base Table (offloading_vs_loading):")
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    item_code, 
                    qty_loaded, 
                    qty_offloaded, 
                    price,
                    value_loaded,
                    value_offloaded,
                    diff_val
                FROM offloading_vs_loading
                WHERE item_code = '06587'
                LIMIT 3
            """)
            rows = cur.fetchall()
            for row in rows:
                print(f"  Item: {row['item_code']}, Qty Loaded: {row['qty_loaded']}, Qty Offloaded: {row['qty_offloaded']}, Price: {row['price']}")
                print(f"    Stored Values - Loaded: {row['value_loaded']}, Offloaded: {row['value_offloaded']}, Diff: {row['diff_val']}")
                
        # Check MV
        print("\n📊 Materialized View (mv_ovl_detail_all):")
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    item_code, 
                    qty_loaded, 
                    qty_offloaded, 
                    price,
                    value_loaded,
                    value_offloaded,
                    diff_val
                FROM mv_ovl_detail_all
                WHERE item_code = '06587'
                LIMIT 3
            """)
            rows = cur.fetchall()
            for row in rows:
                print(f"  Item: {row['item_code']}, Qty Loaded: {row['qty_loaded']}, Qty Offloaded: {row['qty_offloaded']}, Price: {row['price']}")
                print(f"    Computed Values - Loaded: {row['value_loaded']}, Offloaded: {row['value_offloaded']}, Diff: {row['diff_val']}")
                print(f"    ✓ {row['qty_loaded']} * {row['price']} = {row['value_loaded']}")
                print(f"    ✓ {row['qty_offloaded']} * {row['price']} = {row['value_offloaded']}")
                print(f"    ✓ ({row['qty_offloaded']} - {row['qty_loaded']}) * {row['price']} = {row['diff_val']}")
        
        conn.close()
    except psycopg2.Error as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    verify_item()
