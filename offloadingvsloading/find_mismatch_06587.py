#!/usr/bin/env python3
"""Find item 06587 rows with different loaded/offloaded quantities."""
import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'dbname': 'WH'
}

def find_mismatch():
    """Find rows where qty_loaded != qty_offloaded for item 06587."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        print("📊 Looking for rows with mismatched quantities (from 10 to 20):")
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    date,
                    shop_code,
                    item_code,
                    qty_loaded,
                    qty_offloaded,
                    price,
                    value_loaded,
                    value_offloaded,
                    diff_val
                FROM offloading_vs_loading
                WHERE item_code = '06587'
                ORDER BY date DESC, shop_code
            """)
            rows = cur.fetchall()
            if not rows:
                print("  ❌ No rows found for item 06587")
            else:
                for i, row in enumerate(rows[:10]):  # Show first 10 rows
                    print(f"\n  Row {i+1} ({row['date']} | {row['shop_code']}):")
                    print(f"    Qty Loaded: {row['qty_loaded']}, Qty Offloaded: {row['qty_offloaded']}, Price: {row['price']}")
                    print(f"    Stored - Loaded Value: {row['value_loaded']}, Offloaded Value: {row['value_offloaded']}, Diff: {row['diff_val']}")
                    print(f"    Expected - Loaded: {row['qty_loaded']*row['price'] if row['price'] else 0}, Offloaded: {row['qty_offloaded']*row['price'] if row['price'] else 0}, Diff: {(row['qty_offloaded']-row['qty_loaded'])*row['price'] if row['price'] else 0}")
        
        conn.close()
    except psycopg2.Error as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    find_mismatch()
